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

    def to_json(self):
        return json.dumps({
            "mode": self.mode,
            "total_segments": self.total_segments,
            "resolution": self.resolution,
            "target_fps": self.target_fps,
            "backend": self.backend,
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
            mode=data.get("mode", "one_shot"),
            total_segments=data.get("total_segments", len(segments)),
            resolution=data.get("resolution", [832, 480]),
            target_fps=data.get("target_fps", 16),
            segments=segments,
            backend=data.get("backend", "wanvideo"),
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

    def to_dict(self):
        return {
            "segment_index": self.segment_index,
            "video_path": self.video_path,
            "last_frame_path": self.last_frame_path,
            "status": self.status,
            "prompt_id": self.prompt_id,
            "error": self.error,
            "duration_sec": self.duration_sec,
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

SEGMENT_MODE_ONE_SHOT = "one_shot"
SEGMENT_MODE_SMART_SPLIT = "smart_split"
SEGMENT_MODE_SLIDING_WINDOW = "sliding_window"

# 生成后端标识（SegmentPlan.backend）
BACKEND_WANVIDEO = "wanvideo"   # 骨骼路线：4k+1 帧规则
BACKEND_SCAIL2 = "scail2"       # SCAIL-2 路线：每段 81 帧、重叠 5、有效步进 76

REF_STRATEGY_USER_IMAGE = "user_image"
REF_STRATEGY_PREV_LAST_FRAME = "prev_last_frame"
REF_STRATEGY_AUTO_SELECT = "auto_select"
