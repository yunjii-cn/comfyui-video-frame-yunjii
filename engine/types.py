import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SegmentInfo:
    index: int
    start_frame: int
    end_frame: int
    target_frames: int
    overlap_prev: int = 0
    overlap_next: int = 0
    complexity: float = 0.5
    ref_strategy: str = "user_image"
    prompt: str = ""
    negative_prompt: str = ""
    params: dict = field(default_factory=lambda: {
        "steps": 30,
        "cfg": 6.0,
        "denoise": 1.0,
    })

    def to_dict(self):
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "target_frames": self.target_frames,
            "overlap_prev": self.overlap_prev,
            "overlap_next": self.overlap_next,
            "complexity": self.complexity,
            "ref_strategy": self.ref_strategy,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "params": self.params,
        }


@dataclass
class SegmentPlan:
    mode: str
    total_segments: int
    resolution: list = field(default_factory=lambda: [832, 480])
    target_fps: int = 16
    segments: list = field(default_factory=list)
    effects: list = field(default_factory=list)
    backend: str = "wanvideo"  # BACKEND_WANVIDEO / BACKEND_SCAIL2
    # 一镜到底单次超长生成(方案C)：True=单段覆盖驱动视频全长，不切段不拼接，
    # 由 WanVideoContextOptions 滑窗覆盖全帧，pose_latent 覆盖全帧，单次连贯去噪、零转场。
    single_pass: bool = False
    # 连贯策略（生成侧时序连续性方案）：multi_seg / single_pass / warm_start
    continuity_strategy: str = "multi_seg"
    # 无缝连贯方案档位：seamless_A / seamless_B / seamless_C / seamless_auto
    # A=标准多段无缝(一般时长)；B=超长视频无缝(长程防漂移,启用 long_video_mode)；
    # C=单遍连贯(旧方案C兜底)；auto=按连贯策略归一(兼容旧工作流)。
    seamless_plan: str = "seamless_auto"
    # 超长视频模式（B 方案启用）：True 时加大重叠冗余、强制真骨架防漂移、段数按容量上限分块，
    # 抑制 15~30s+ 长视频的长程姿态/光影漂移累积。
    long_video_mode: bool = False
    # 模型精度：fp8(默认,省显存) / fp16(更精细,吃显存) —— 仅 SCAIL-2 路线生效
    model_precision: str = "fp8"
    # 单遍时长上限(秒)：>0 时限制方案C单遍长度，超出则回退多段seamless，抑制长程稀释画质退化
    single_pass_cap: float = 0.0

    def to_json(self):
        return json.dumps({
            "mode": self.mode,
            "total_segments": self.total_segments,
            "resolution": self.resolution,
            "target_fps": self.target_fps,
            "backend": self.backend,
            "single_pass": self.single_pass,
            "continuity_strategy": self.continuity_strategy,
            "seamless_plan": self.seamless_plan,
            "long_video_mode": self.long_video_mode,
            "model_precision": self.model_precision,
            "single_pass_cap": self.single_pass_cap,
            "segments": [s.to_dict() if isinstance(s, SegmentInfo) else s for s in self.segments],
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        segments = []
        for s in data.get("segments", []):
            if isinstance(s, dict):
                segments.append(SegmentInfo(**{k: v for k, v in s.items() if k in SegmentInfo.__dataclass_fields__}))
            else:
                segments.append(s)
        return cls(
            mode=data.get("mode", SEGMENT_MODE_ONE_SHOT),
            total_segments=data.get("total_segments", len(segments)),
            resolution=data.get("resolution", [832, 480]),
            target_fps=data.get("target_fps", 16),
            segments=segments,
            backend=data.get("backend", "wanvideo"),
            single_pass=data.get("single_pass", False),
            continuity_strategy=data.get("continuity_strategy", "multi_seg"),
            seamless_plan=data.get("seamless_plan", "seamless_auto"),
            long_video_mode=data.get("long_video_mode", False),
            model_precision=data.get("model_precision", "fp8"),
            single_pass_cap=data.get("single_pass_cap", 0.0),
        )


@dataclass
class SegmentResult:
    segment_index: int
    video_path: str = ""
    last_frame_path: str = ""
    status: str = "pending"
    prompt_id: str = ""
    error: str = ""
    duration_sec: float = 0.0
    # 与前一段的重叠帧数（一镜到底链式生成时 >0）。拼接阶段据此去重重叠帧，实现零转场。
    overlap_prev: int = 0
    # 潜空间拼接：本段 latent 落盘路径（STITCH_LATENT_BLEND 模式用于合并+解码）。
    # 由 runner 按 run_id/seg 索引确定性构造（与适配器 _latent_save_path 一致）。
    latent_path: str = ""

    def to_dict(self):
        return {
            "segment_index": self.segment_index,
            "video_path": self.video_path,
            "last_frame_path": self.last_frame_path,
            "status": self.status,
            "prompt_id": self.prompt_id,
            "error": self.error,
            "duration_sec": self.duration_sec,
            "overlap_prev": self.overlap_prev,
            "latent_path": self.latent_path,
        }


@dataclass
class SegmentContext:
    last_frame_path: str = ""
    color_palette: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class NodeMap:
    animate_embeds: str = ""
    video_combine: str = ""
    ref_image: str = ""
    ref_video: str = ""
    sampler: str = ""
    text_encode: str = ""
    pose_images: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v}

    def is_valid(self):
        return bool(self.animate_embeds and self.sampler and self.text_encode)


# ⚠️ 拼接模式本质澄清（2026-08-12 用户反馈后纠正）：
#   用户实测并明确指出——以下三种**事后混合**模式，无论像素层还是 latent 层，
#   本质都只是在「两段视频的两端之间做转场/淡化」，**不是真正的连贯无缝拼接**：
#     · STITCH_LATENT_BLEND   = 潜空间 latent 交叉淡化（= latent 层面的淡入淡出转场）
#     · STITCH_SEAMLESS_BLEND = 段间像素级短窗交叉溶解
#     · STITCH_CROSS_DISSOLVE = 像素交叉淡化
#   真正的「无感一镜到底」必须**在生成侧就让多段天然连续**（context_options 滑动窗口 +
#   跨段 reference_latent 续写），拼接阶段只需 STITCH_SEAMLESS 硬切丢重叠帧即可看不出接缝。
#   故：本机推荐拼接模式 = STITCH_SEAMLESS（零混合硬切），配合上方『无缝连贯方案』使用。
STITCH_HARD_CUT = "hard_cut"
STITCH_CROSS_DISSOLVE = "cross_dissolve"
STITCH_LATENT_BLEND = "latent_blend"
STITCH_TRANSITION = "transition"
STITCH_AUTO = "auto"
# 真·一镜到底（零转场）：硬切 + 丢弃后续段头部的重叠帧（去重），零混合。
# 与「无缝连贯方案（生成侧连续）」配合时，段边界本身连续，硬切即看不出接缝。
# 注意：若生成侧未做连续（旧多段独立生成），硬切反而会暴露跳变——此时才是上三种混合的「兜底化妆」。
STITCH_SEAMLESS = "seamless"
# 兼容别名：旧 UI 里称为「重叠混合」的拼接模式，本质仍是转场（见上方澄清），保留以防旧工作流断链。
STITCH_SEAMLESS_BLEND = "seamless_blend"

# —— 拼接模式：中文标签（下拉显示） ↔ 英文值（内部逻辑/旧工作流存值） ——
# 显示用中文，比较/存储仍用英文值；旧已保存的英文值（auto/transition...）仍兼容。
# 📌 标签后缀标注 [转场] 以直观提示用户：凡标 [转场] 的都是事后混合、非真无缝。
STITCH_LABELS = [
    (STITCH_SEAMLESS,       "无缝一镜到底(零转场·硬切) ⭐推荐"),
    (STITCH_HARD_CUT,       "硬切"),
    (STITCH_AUTO,           "自动"),
    (STITCH_SEAMLESS_BLEND, "无缝一镜到底(重叠混合)[转场]"),
    (STITCH_LATENT_BLEND,   "真·一镜到底（潜空间拼接）[转场]"),
    (STITCH_CROSS_DISSOLVE, "交叉淡化[转场]"),
    (STITCH_TRANSITION,     "ffmpeg转场"),
]
STITCH_LABEL_TO_VALUE = {label: value for value, label in STITCH_LABELS}
STITCH_LABEL_TO_VALUE.update({value: value for value, _ in STITCH_LABELS})  # 旧英文值也能归一

SEGMENT_MODE_ONE_SHOT = "一镜到底"
SEGMENT_MODE_SMART_SPLIT = "智能分段"
SEGMENT_MODE_SLIDING_WINDOW = "滑动窗口"

# —— 连贯策略（生成侧时序连续性方案，与拼接模式正交）—— #
# 多段无缝：分段独立 I2V + 接缝混合（仅化妆，默认）
# 单遍连贯：整片一次去噪(latent连续)，方案C，长视频画质软
# 暖启动：分段 + 上段真实帧喂回 WanAnimatePlus prefix_frames（Tier2，连续+画质）
CONTINUITY_MULTI_SEG = "multi_seg"
CONTINUITY_SINGLE_PASS = "single_pass"
CONTINUITY_WARM_START = "warm_start"
CONTINUITY_AUTO = "auto"  # 兼容旧 单遍连贯模式 bool 的未显式选择态

CONTINUITY_LABELS = [
    (CONTINUITY_MULTI_SEG,   "多段无缝(默认)"),
    (CONTINUITY_SINGLE_PASS, "单遍连贯(方案C)"),
    (CONTINUITY_WARM_START,  "暖启动(Tier2)"),
]
CONTINUITY_LABEL_TO_VALUE = {label: value for value, label in CONTINUITY_LABELS}
CONTINUITY_LABEL_TO_VALUE.update({value: value for value, _ in CONTINUITY_LABELS})

# 生成后端标识（SegmentPlan.backend）
BACKEND_WANVIDEO = "wanvideo"   # 骨骼路线：4k+1 帧规则
BACKEND_SCAIL2 = "scail2"       # SCAIL-2 路线：每段 81 帧、重叠 5、有效步进 76

REF_STRATEGY_USER_IMAGE = "user_image"
REF_STRATEGY_PREV_LAST_FRAME = "prev_last_frame"
REF_STRATEGY_AUTO_SELECT = "auto_select"


# —— 无缝连贯方案（生成侧时序连续性档位，用户可选 A / B / C）——
# 三者**共用**同一套真·无缝机制：WanVideoContextOptions 滑窗(context_overlap=32=8 latent 帧)
#   + 跨段 reference_latent 续写（上段 latent 喂下一段上下文窗口），使多段在同一条去噪轨迹连续，
#   拼接阶段只需 STITCH_SEAMLESS 零转场硬切即可无感衔接。互不冲突，区别仅在目标时长与防漂移增强。
#
#   A 方案（标准多段无缝）：一般时长(≤约15s)多段连续生成。每段独立高保真，接缝靠生成侧连续消灭。
#       优点：每段质量最高、显存友好、每段独立重试。  缺点：超长视频(>15~20s)段数多，长程姿态/光影漂移累积。
#   B 方案（超长视频无缝）：同样机制，但启用 long_video_mode——加大重叠冗余、强制 SCAIL 真骨架(逐帧姿态)
#       抑制长程漂移、段数自动按容量上限分块(≤~9 段×81帧≈30s+)，并可选末端闭环校正。
#       优点：15~30s+ 长视频无断点、无质量降级，最适合「一镜到底长片」。  缺点：段数多→总耗时线性增加；
#       若某段失败需重跑该段(已有机制)。  与 MiniMax H3 的「分段参考延长(Ref2VA)」原理同源，但兼容本机 SCAIL-2。
#   C 方案（单遍连贯·旧方案C）：整片一次去噪(latent 天然连续)，真·一镜到底。仅作兜底/对比，不推荐主用：
#       优点：零分段、一次去噪最连贯理论值。  缺点：长视频被长程重度混合稀释→画质软、显存峰值高、不可分段重试。
#
# 注意：A/B 是主推互补路线（A 用于短、B 用于长），C 仅备选。三者都与「连贯策略(Warm_start/Tier2)」正交，
#   Tier2 像素 prefix 冻结已证实弱于本机制，故本下拉默认即取代 Tier2 成为无缝主方案。
SEAMLESS_PLAN_A = "seamless_A"        # 标准多段无缝（一般时长）
SEAMLESS_PLAN_B = "seamless_B"        # 超长视频无缝（长程防漂移增强）
SEAMLESS_PLAN_C = "seamless_C"        # 单遍连贯（旧方案C，兜底）
SEAMLESS_PLAN_AUTO = "seamless_auto"  # 兼容旧：按连贯策略归一

SEAMLESS_PLAN_LABELS = [
    (SEAMLESS_PLAN_A, "A方案·标准多段无缝(一般时长≤15s) ⭐默认"),
    (SEAMLESS_PLAN_B, "B方案·超长视频无缝(15~30s+防漂移)"),
    (SEAMLESS_PLAN_C, "C方案·单遍连贯(旧方案C·兜底)"),
]
SEAMLESS_PLAN_LABEL_TO_VALUE = {label: value for value, label in SEAMLESS_PLAN_LABELS}
SEAMLESS_PLAN_LABEL_TO_VALUE.update({value: value for value, _ in SEAMLESS_PLAN_LABELS})

