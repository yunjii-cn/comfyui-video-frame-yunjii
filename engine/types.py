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


STITCH_HARD_CUT = "hard_cut"
STITCH_CROSS_DISSOLVE = "cross_dissolve"
STITCH_LATENT_BLEND = "latent_blend"
STITCH_TRANSITION = "transition"
STITCH_AUTO = "auto"
# 真·一镜到底：硬切 + 丢弃后续段头部的重叠帧（去重），零转场、零重复帧
STITCH_SEAMLESS = "seamless"
# 一镜到底重叠混合：在段间接缝处做短窗（淡化帧数）交叉溶解，软化位置跳变；
# 注意：仅缓解位置不连续，无法修复动作(速度)跳变——真·连贯需生成侧前帧时序上下文(see 记忆)。
STITCH_SEAMLESS_BLEND = "seamless_blend"

# —— 拼接模式：中文标签（下拉显示） ↔ 英文值（内部逻辑/旧工作流存值） ——
# 显示用中文，比较/存储仍用英文值；旧已保存的英文值（auto/transition...）仍兼容。
STITCH_LABELS = [
    (STITCH_HARD_CUT,       "硬切"),
    (STITCH_CROSS_DISSOLVE, "交叉淡化"),
    (STITCH_AUTO,           "自动"),
    (STITCH_SEAMLESS,       "无缝一镜到底(零转场)"),
    (STITCH_SEAMLESS_BLEND, "无缝一镜到底(重叠混合)"),
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
