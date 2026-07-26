import logging

from .base import EffectModule, EffectContext
from ..types import (
    SEGMENT_MODE_ONE_SHOT,
    REF_STRATEGY_PREV_LAST_FRAME, REF_STRATEGY_AUTO_SELECT,
)

logger = logging.getLogger(__name__)


class MimicEffect:
    def __init__(self, mode="full_mimic"):
        self._mode = mode

    @property
    def stage(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return f"mimic_{self._mode}"

    def transform_poses(self, poses, context):
        if self._mode == "full_mimic":
            return poses
        if self._mode == "motion_transfer":
            return poses
        if self._mode == "person_swap":
            return poses
        if self._mode == "style_transfer":
            return poses
        return poses

    def transform_prompts(self, prompts, context):
        if self._mode == "full_mimic":
            return prompts
        if self._mode == "motion_transfer":
            enhanced = []
            for p in prompts:
                if "in " in p.lower() or "at " in p.lower():
                    enhanced.append(p)
                else:
                    enhanced.append(p)
            return enhanced
        return prompts

    def transform_params(self, params, context):
        if self._mode == "full_mimic":
            params["ref_strategy"] = REF_STRATEGY_PREV_LAST_FRAME
        elif self._mode == "person_swap":
            params["ref_strategy"] = REF_STRATEGY_AUTO_SELECT
        return params

    def transform_stitch(self, stitch_plan, context):
        if self._mode == "full_mimic":
            stitch_plan["mode"] = "cross_dissolve"
        return stitch_plan
