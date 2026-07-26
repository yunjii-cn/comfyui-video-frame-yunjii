import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EffectModule(Protocol):
    @property
    def stage(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return ""

    def transform_poses(self, poses, context):
        return poses

    def transform_prompts(self, prompts, context):
        return prompts

    def transform_params(self, params, context):
        return params

    def transform_stitch(self, stitch_plan, context):
        return stitch_plan


class EffectContext:
    def __init__(self, video_analysis=None, pose_sequence=None, prompts=None,
                 params=None, stitch_plan=None, metadata=None):
        self.video_analysis = video_analysis
        self.pose_sequence = pose_sequence
        self.prompts = prompts or []
        self.params = params or {}
        self.stitch_plan = stitch_plan
        self.metadata = metadata or {}
