import logging

from .base import EffectModule, EffectContext
from ..types import STITCH_AUTO, STITCH_CROSS_DISSOLVE

logger = logging.getLogger(__name__)


# 身份一致性后缀：强化「角色外观恒定」的语义。
# 注意：身份在模型侧已由 SCAIL-2/WanVideo 的参考图恒定注入 + previous_frames 串联保证，
# 这里是 prompt 侧的互补增强（不是重复），对「长视频完美模仿」的跨段一致性是纯增益。
IDENTITY_SUFFIX = (
    "consistent character identity, same clothing and face, "
    "stable appearance and lighting across all shots"
)


class MimicModule(EffectModule):
    """模仿增强模块 —— P3-B 的第一个真实效果模块。

    它证明「效果管线」是可用、可消费的，而不是空接口：
    - 生成侧：在非首段 prompt 追加身份一致性后缀（首段本身就是参考，无需强调）；
    - 拼接侧：交叉淡化 / 自动模式下把淡化帧数抬高，让身份在段间过渡更平滑；
    - 参数侧：默认透传（不擅自改用户的 steps/cfg，避免引入非预期回归）。

    空效果模块列表时本模块不会被实例化，runner/stitcher 行为完全不变（零回归）。
    """

    stage = 10
    name = "mimic"

    def __init__(self, reinforce_prompt: bool = True, prompt_suffix: str = None,
                 min_fade_frames: int = 12):
        self.reinforce_prompt = reinforce_prompt
        self.prompt_suffix = prompt_suffix or IDENTITY_SUFFIX
        self.min_fade_frames = min_fade_frames

    def transform_prompts(self, prompts, context: EffectContext):
        if not self.reinforce_prompt or not prompts:
            return prompts
        out = []
        for i, p in enumerate(prompts):
            p = p or ""
            if i == 0:
                # 首段即参考来源，强调身份反而可能限制生成，保持原样
                out.append(p)
                continue
            sep = ", " if p.strip() else ""
            out.append(f"{p}{sep}{self.prompt_suffix}")
        return out

    def transform_params(self, params, context: EffectContext):
        # 默认透传：参数（steps/cfg/denoise）由用户/planner 决定，
        # 本模块不擅自覆盖，保证「选 mimic 时行为与现状等价」。
        return params

    def transform_stitch(self, stitch_plan, context: EffectContext):
        if not stitch_plan:
            return stitch_plan
        mode = stitch_plan.get("mode")
        if mode in ("auto", "cross_dissolve", STITCH_AUTO, STITCH_CROSS_DISSOLVE):
            sp = dict(stitch_plan)
            sp["fade_frames"] = max(int(stitch_plan.get("fade_frames", 8)), self.min_fade_frames)
            return sp
        return stitch_plan
