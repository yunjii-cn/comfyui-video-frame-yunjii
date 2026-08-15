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


# ⚠️ 拼接模式本质澄清（2026-08-15 用户实测反馈后再次纠正）：
#   用户实测并明确指出——以下三种**事后混合**模式，无论像素层还是 latent 层，
#   本质都只是在「两段视频的两端之间做转场/淡化」，**不是真正的连贯无缝拼接**：
#     · STITCH_LATENT_BLEND   = 潜空间 latent 交叉淡化（= latent 层面的淡入淡出转场）
#     · STITCH_SEAMLESS_BLEND = 段间像素级短窗交叉溶解
#     · STITCH_CROSS_DISSOLVE = 像素交叉淡化
#   而且实测证实：生成侧「跨段 reference_latent 续写」在本节点里本质只是 i2v 首帧弱偏置，
#   被 4 步蒸馏洗掉、名不副实，多段仍不连贯。故真无缝不能只靠生成侧。
#   真正可靠的真无缝 = 拼接阶段的「帧锚定」：丢弃生成侧重叠头帧后，**硬把下一段首帧
#   替换为上一段尾帧(像素级相等)**，再短窗交叉淡化软化 → 100% 保证「下一段首帧=上一段尾帧」，
#   段边界像素级连续、硬切即无缝，且可离线验证。对应 STITCH_FRAME_ANCHOR（⭐推荐）。
#   STITCH_SEAMLESS 现等价帧锚定（兼容旧标签）；自动模式在 SCAIL 路线默认即走帧锚定。
STITCH_HARD_CUT = "hard_cut"
STITCH_CROSS_DISSOLVE = "cross_dissolve"
STITCH_LATENT_BLEND = "latent_blend"
STITCH_TRANSITION = "transition"
STITCH_AUTO = "auto"
# 真·一镜到底（零转场）：硬切 + 丢弃后续段头部的重叠帧（去重），零混合。
# 与「无缝连贯方案（生成侧连续）」配合时，段边界本身连续，硬切即看不出接缝。
# 注意：若生成侧未做连续（旧多段独立生成），硬切反而会暴露跳变——此时才是上三种混合的「兜底化妆」。
STITCH_SEAMLESS = "seamless"
# 帧锚定一镜到底（⭐推荐真无缝）：拼接阶段确定性像素锚定——丢弃生成侧重叠头帧后，
# 硬把下一段首帧替换为上一段尾帧(像素级相等)，再向后做短窗交叉淡化软化。
# 与依赖生成侧 reference_latent 续写(被 4 步蒸馏洗掉、名不副实)不同，本模式在拼接阶段
# 100% 保证「下一段首帧=上一段尾帧」，段边界像素级连续、硬切即无缝，且可离线验证。
STITCH_FRAME_ANCHOR = "frame_anchor"
# 兼容别名：旧 UI 里称为「重叠混合」的拼接模式，本质仍是转场（见上方澄清），保留以防旧工作流断链。
STITCH_SEAMLESS_BLEND = "seamless_blend"

# —— 拼接模式：中文标签（下拉显示） ↔ 英文值（内部逻辑/旧工作流存值） ——
# 显示用中文，比较/存储仍用英文值；旧已保存的英文值（auto/transition...）仍兼容。
# 📌 标签后缀标注 [转场] 以直观提示用户：凡标 [转场] 的都是事后混合、非真无缝。
STITCH_LABELS = [
    # 自动放首位并作默认：跟随后端/连贯方案自动选最优（详见 composer._continuity_capable 分支），
    # 用户在「分段规划」选完 A/B/暖启动后，这里保持「自动」即可，不必再选第二次。
    (STITCH_AUTO,           "自动(跟随方案最优)"),
    # ⭐推荐真无缝：拼接阶段确定性像素锚定(下一段首帧=上一段尾帧)，不依赖生成侧 latent 链。
    (STITCH_FRAME_ANCHOR,   "帧锚定一镜到底(首帧=尾帧)⭐"),
    # 零转场硬切：现同样在拼接阶段做首帧锚定(等价于帧锚定)，保留作兼容性标签。
    (STITCH_SEAMLESS,       "无缝一镜到底(零转场·首帧锚定)"),
    (STITCH_CROSS_DISSOLVE, "交叉淡化[转场]"),
    (STITCH_SEAMLESS_BLEND, "无缝一镜到底(重叠混合)[转场]"),
    # 旧称「真·一镜到底（潜空间拼接）⭐推荐」是误导——它本质是 latent 层交叉淡化(转场)，并非真无缝；改名并去⭐。
    (STITCH_LATENT_BLEND,   "潜空间交叉淡化(转场)"),
    (STITCH_HARD_CUT,       "硬切"),
    (STITCH_TRANSITION,     "ffmpeg转场"),
]
STITCH_LABEL_TO_VALUE = {label: value for value, label in STITCH_LABELS}
STITCH_LABEL_TO_VALUE.update({value: value for value, _ in STITCH_LABELS})  # 旧英文值也能归一
# 旧版标签 → 当前值：改名前的误导标签串需仍能归一，避免旧工作流静默回退默认。
_LEGACY_STITCH_LABELS = {
    "真·一镜到底（潜空间拼接）⭐推荐[转场]": STITCH_LATENT_BLEND,
    "无缝一镜到底(零转场·硬切)":           STITCH_SEAMLESS,  # 曾用同名微调，稳妥兼容
    "自动":                               STITCH_AUTO,  # 旧「自动」标签仍归一，避免旧工作流断链
}
STITCH_LABEL_TO_VALUE.update(_LEGACY_STITCH_LABELS)

# 拼接模式默认项（下拉首项「自动(跟随方案最优)」），单一入口跟随后端选最优，用户无需二次选择。
STITCH_DEFAULT = STITCH_AUTO

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
BACKEND_SCAIL2_NATIVE = "scail2_native"  # 原生 SCAIL-2 长视频节点(一镜到底动作模仿)：comfyui_scail2_multi_cond

REF_STRATEGY_USER_IMAGE = "user_image"
REF_STRATEGY_PREV_LAST_FRAME = "prev_last_frame"
REF_STRATEGY_AUTO_SELECT = "auto_select"


# —— 无缝连贯方案（生成侧时序连续性档位，用户可选 A / B / C）——
# 三者**共用**同一套真·无缝机制：WanVideoContextOptions 滑窗(context_overlap=32=8 latent 帧)
#   + 跨段 reference_latent 续写（上段 latent 喂下一段上下文窗口），使多段在同一条去噪轨迹连续，
#   拼接阶段只需 STITCH_SEAMLESS 零转场硬切即可无感衔接。互不冲突，区别仅在目标时长与防漂移增强。
#
#   A 方案（标准多段无缝）：一般时长(≤约15s)多段独立生成，接缝由 composer 交叉溶解平滑过渡。
#       优点：每段质量最高、显存友好、每段独立可重试。  缺点：独立去噪、段边界为平滑过渡(非真连续)，
#       超长视频(>15~20s)段数多会累积长程姿态/光影漂移。
#   B 方案（超长视频无缝）：单遍连续采样 + context 滑窗覆盖全帧(81帧一窗/重叠32潜空间fuse)，
#       整片作为一条去噪轨迹 → 真·无漂移、真无缝，长视频(15~30s+) 无劣化（⭐长片推荐）。
#       优点：真·零接缝、长视频不崩。  缺点：显存峰值更高、不可分段重试、总耗时随总长线性增长。
#       与 C 同为 single_pass 规划，但 B 注入 context 滑窗(真无缝)、C 不注入(旧兜底, >5s 画质软)。
#   C 方案（单遍连贯·旧方案C兜底）：整片一次去噪、不注入 context 滑窗 → >5s 画质软，仅作对比/兜底，不推荐主用。
#       优点：零分段、一次去噪。  缺点：长视频被长程稀释→画质软、显存峰值高、不可分段重试。
#
# 注意：A/B 是主推互补路线（A 用于短、B 用于长），C 仅备选。三者都与「连贯策略(Warm_start/Tier2)」正交，
#   Tier2 像素 prefix 冻结已证实弱于本机制，故本下拉默认即取代 Tier2 成为无缝主方案。
SEAMLESS_PLAN_A = "seamless_A"        # 标准多段无缝（一般时长）
SEAMLESS_PLAN_B = "seamless_B"        # 超长视频无缝（长程防漂移增强）
SEAMLESS_PLAN_C = "seamless_C"        # 单遍连贯（旧方案C，兜底）
SEAMLESS_PLAN_AUTO = "seamless_auto"  # 兼容旧：按连贯策略归一
SEAMLESS_PLAN_SMART_SPLIT = "seamless_smart_split"  # 智能分段(转场编排)：独立分段+重叠混合过渡

SEAMLESS_PLAN_LABELS = [
    (SEAMLESS_PLAN_A, "A方案·标准多段无缝(独立生成+平滑过渡, ≤15s) ⭐默认"),
    (SEAMLESS_PLAN_B, "B方案·超长视频无缝(单遍滑窗真·无缝, 15~30s+ ⭐长片推荐)"),
    (SEAMLESS_PLAN_C, "C方案·单遍兜底(不滑窗, >5s画质软)"),
]
SEAMLESS_PLAN_LABEL_TO_VALUE = {label: value for value, label in SEAMLESS_PLAN_LABELS}
SEAMLESS_PLAN_LABEL_TO_VALUE.update({value: value for value, _ in SEAMLESS_PLAN_LABELS})


# —— 统一「连贯方案」下拉：合并旧 生成模式 / 单遍连贯模式(bool) / 连贯策略 / 无缝连贯方案 为一组 ——
# 用户只面对一个选择：A 标准多段无缝 / B 超长视频无缝 / C 单遍兜底 / 暖启动(Tier2) / 智能分段(转场编排)。
#   每个选项自洽地派生 (strategy, seamless_plan, mode)，不再有隐藏闸门或孤儿选项：
#   A → multi_seg + seamless_A + mode=一镜到底（每段独立+硬切去重零转场，≤15s，质量最高、可分段重试）
#   B → single_pass(仅一镜到底+SCAIL2) + seamless_B + mode=一镜到底（单遍滑窗真·无缝，15~30s+ 不劣化 ⭐长片推荐）
#   C → single_pass + seamless_C + mode=一镜到底（不滑窗，>5s画质软，仅对比/兜底）
#   暖启动 → warm_start + seamless_auto + mode=一镜到底（WanAnimatePlus 多段+上段真实帧喂回 prefix_frames）
#   智能分段(转场编排) → multi_seg + seamless_A + mode=智能分段（独立分段+重叠混合过渡，明确要转场时用，非一镜到底）
# 旧「生成模式」中文/英文值(一镜到底/智能分段/滑动窗口/one_shot/smart_split/sliding_window)由 resolve 归一：
#   一镜到底/one_shot→A；智能分段/smart_split→智能分段；滑动窗口/sliding_window→B（滑窗无缝）。
UNIFIED_PLAN_DEFAULT = SEAMLESS_PLAN_A
# 显示名设计原则（面向小白）：① 以「身份词」开头，5 项一眼可区分；② 紧跟用途(时长/场景)；
# ③ 末尾点明关键特征或代价；④ 默认/推荐用 ⭐ 标注。内部值(SEAMLESS_PLAN_*)保持不变以兼容旧工作流。
UNIFIED_PLAN_LABELS = [
    (SEAMLESS_PLAN_A,           "短视频·多段无缝（≤15秒，每段画质最好）⭐默认"),
    (SEAMLESS_PLAN_B,           "长视频·单遍真无缝（15~30秒+，全程不断裂不劣化）"),
    (SEAMLESS_PLAN_C,           "兜底·单遍旧方案（长视频画质偏软，不推荐）"),
    (CONTINUITY_WARM_START,     "暖启动·帧续写（上一段末帧接下一段，WanAnimatePlus）"),
    (SEAMLESS_PLAN_SMART_SPLIT, "分段转场·重叠混合（每段独立，适合做转场效果）"),
]
UNIFIED_PLAN_LABEL_TO_VALUE = {label: value for value, label in UNIFIED_PLAN_LABELS}
UNIFIED_PLAN_LABEL_TO_VALUE.update({value: value for value, _ in UNIFIED_PLAN_LABELS})

# 旧版下拉标签 → 当前值：旧工作流 widgets_values 里存的是改名前的标签串，需仍能正确归一，
# 否则会静默回退到默认 A。覆盖历次改名（A·/A方案· 前缀、Tier2、转场编排 等写法）。
_LEGACY_UNIFIED_LABELS = {
    "A·标准多段无缝(≤15s) ⭐默认":                          SEAMLESS_PLAN_A,
    "A方案·标准多段无缝(独立生成+平滑过渡, ≤15s) ⭐默认":     SEAMLESS_PLAN_A,
    "B·超长视频无缝(15~30s+ 单遍滑窗真·无缝)":              SEAMLESS_PLAN_B,
    "B方案·超长视频无缝(单遍滑窗真·无缝, 15~30s+ ⭐长片推荐)": SEAMLESS_PLAN_B,
    "C·单遍兜底(不滑窗, >5s画质软)":                        SEAMLESS_PLAN_C,
    "C方案·单遍兜底(不滑窗, >5s画质软)":                    SEAMLESS_PLAN_C,
    "暖启动(Tier2)·WanAnimatePlus上段末帧续写":             CONTINUITY_WARM_START,
    "智能分段(转场编排)·独立分段+重叠混合过渡":              SEAMLESS_PLAN_SMART_SPLIT,
    "智能分段(转场编排)":                                   SEAMLESS_PLAN_SMART_SPLIT,
}
UNIFIED_PLAN_LABEL_TO_VALUE.update(_LEGACY_UNIFIED_LABELS)


# 旧「生成模式」(独立下拉时代) 中文/英文值 → 对应统一方案值。
# 一镜到底/one_shot → A；智能分段/smart_split → 智能分段(转场编排)；
# 滑动窗口/sliding_window → B（滑窗无缝，旧孤儿选项在此获得真实语义）。
_LEGACY_MODE_TO_UNIFIED = {
    SEGMENT_MODE_ONE_SHOT:       SEAMLESS_PLAN_A,
    "one_shot":                  SEAMLESS_PLAN_A,
    SEGMENT_MODE_SMART_SPLIT:    SEAMLESS_PLAN_SMART_SPLIT,
    "smart_split":               SEAMLESS_PLAN_SMART_SPLIT,
    SEGMENT_MODE_SLIDING_WINDOW: SEAMLESS_PLAN_B,
    "sliding_window":            SEAMLESS_PLAN_B,
}


def resolve_unified_plan(value):
    """把统一『连贯方案』下拉值解析为 (strategy, seamless_plan, mode)。

    - strategy: 生成侧时序连续性方案(multi_seg/single_pass/warm_start)；
                A/B/C 返回 None 表示由调用方结合 backend/mode 继续推导。
    - seamless_plan: A/B/C/auto，供下游适配器/拼接识别。
    - mode: 派生的「生成模式」(一镜到底 / 智能分段)，驱动参考链策略与拼接方式。

    归一顺序：① 旧「生成模式」中文/英文值 → 统一方案；② 统一下拉中文标签 → 值；
    ③ 未知值兜底为 A(一镜到底)。故旧工作流残留的 生成模式 值也能正确升级，无冲突。
    """
    # ① 旧「生成模式」值(独立下拉时代) → 对应统一方案
    v = _LEGACY_MODE_TO_UNIFIED.get(value, value)
    # ② 统一下拉的中文标签 → 值
    v = UNIFIED_PLAN_LABEL_TO_VALUE.get(v, v)
    if v == CONTINUITY_WARM_START:
        return CONTINUITY_WARM_START, SEAMLESS_PLAN_AUTO, SEGMENT_MODE_ONE_SHOT
    if v == SEAMLESS_PLAN_SMART_SPLIT:
        return None, SEAMLESS_PLAN_A, SEGMENT_MODE_SMART_SPLIT
    if v == SEAMLESS_PLAN_B:
        return None, SEAMLESS_PLAN_B, SEGMENT_MODE_ONE_SHOT
    if v == SEAMLESS_PLAN_C:
        return None, SEAMLESS_PLAN_C, SEGMENT_MODE_ONE_SHOT
    # 未知 / seamless_A / seamless_auto → 默认 A(一镜到底)
    return None, SEAMLESS_PLAN_A, SEGMENT_MODE_ONE_SHOT

