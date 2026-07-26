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

    def transform_output(self, video_path, context):
        """拼接后作用于最终成片的钩子（增强/超分/插帧在此挂载）。

        默认透传：模块不实现时返回原路径，保证空管线零回归。
        """
        return video_path


class EffectContext:
    def __init__(self, video_analysis=None, pose_sequence=None, prompts=None,
                 params=None, stitch_plan=None, metadata=None):
        self.video_analysis = video_analysis
        self.pose_sequence = pose_sequence
        self.prompts = prompts or []
        self.params = params or {}
        self.stitch_plan = stitch_plan
        self.metadata = metadata or {}
