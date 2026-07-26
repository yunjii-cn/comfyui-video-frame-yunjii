import json
import logging
from typing import List, Optional

from .effects.base import EffectModule, EffectContext
from .effects.mimic import MimicModule
from .effects.enhance import EnhanceModule
from .effects.cinematic import CinematicModule
from .effects.creative import CreativeModule

logger = logging.getLogger(__name__)


# 已知效果模块注册表。runner / stitcher 通过「效果模块」JSON 列表按名字实例化，无需改动调用方。
KNOWN_MODULES = {
    "mimic": MimicModule,
    "enhance": EnhanceModule,
    "cinematic": CinematicModule,   # P3-D
    "creative": CreativeModule,     # P3-E
}


def _instantiate(entry):
    """把列表项（字符串名 或 {name, params} 对象）实例化为模块。失败返回 None。"""
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("module")
        params = entry.get("params") or {}
    else:
        name = str(entry)
        params = {}
    if not name:
        return None
    cls = KNOWN_MODULES.get(name)
    if not cls:
        logger.warning("build_pipeline: 未知效果模块 '%s'，已跳过", name)
        return None
    try:
        return cls(**params) if params else cls()
    except Exception as e:
        logger.warning("build_pipeline: 效果模块 '%s' 实例化失败(参数=%s): %s", name, params, e)
        return None


def build_pipeline(modules_json: str) -> "EffectPipeline":
    """把「效果模块」输入（JSON 列表 / 逗号分隔文本）解析为 EffectPipeline。

    支持两种写法：
      - 简单名：``["mimic", "enhance", "cinematic", "creative"]`` 或 ``"mimic, enhance"``
      - 带参：``[{"name": "enhance", "params": {"upscale_factor": 2, "target_fps": 32}}]``
              ``[{"name": "creative", "params": {"audio_path": "input/ref.mp4", "pacing": "动感"}}]``

    空字符串或空列表 → 空管线（所有 transform 透传，零回归）。
    未知名字 / 参数错误会告警并跳过，不会中断执行。
    """
    if not modules_json or not modules_json.strip():
        return EffectPipeline([])

    entries = []
    try:
        parsed = json.loads(modules_json)
        if isinstance(parsed, list):
            entries = parsed
        elif isinstance(parsed, dict) and "modules" in parsed:
            entries = parsed.get("modules", [])
        else:
            entries = [parsed]
    except Exception:
        # 容错：当作逗号分隔的纯文本列表
        entries = [n.strip() for n in modules_json.split(",") if n.strip()]

    modules = [m for m in (_instantiate(e) for e in entries) if m]
    return EffectPipeline(modules)


class EffectPipeline:
    """按 stage 升序依次应用效果模块。

    无模块时每个 transform 直接返回输入（透传），保证「不选效果模块时输出与现状一致」。
    每个模块的每次 transform 都包了 try/except，单个模块异常不影响整条链路。
    """

    def __init__(self, modules: Optional[List[EffectModule]] = None):
        self.modules = sorted(modules or [], key=lambda m: getattr(m, "stage", 0))

    @property
    def is_empty(self) -> bool:
        return len(self.modules) == 0

    def describe(self) -> str:
        if not self.modules:
            return "(无效果模块)"
        return "; ".join(
            f"{getattr(m, 'name', '?')}@stage{getattr(m, 'stage', 0)}"
            for m in self.modules
        )

    def transform_poses(self, poses, context: EffectContext):
        for m in self.modules:
            try:
                poses = m.transform_poses(poses, context)
            except Exception as e:
                logger.warning("effect %s.transform_poses 失败: %s", getattr(m, "name", "?"), e)
        return poses

    def transform_prompts(self, prompts, context: EffectContext):
        for m in self.modules:
            try:
                prompts = m.transform_prompts(prompts, context)
            except Exception as e:
                logger.warning("effect %s.transform_prompts 失败: %s", getattr(m, "name", "?"), e)
        return prompts

    def transform_params(self, params, context: EffectContext):
        for m in self.modules:
            try:
                params = m.transform_params(params, context)
            except Exception as e:
                logger.warning("effect %s.transform_params 失败: %s", getattr(m, "name", "?"), e)
        return params

    def transform_stitch(self, stitch_plan, context: EffectContext):
        for m in self.modules:
            try:
                stitch_plan = m.transform_stitch(stitch_plan, context)
            except Exception as e:
                logger.warning("effect %s.transform_stitch 失败: %s", getattr(m, "name", "?"), e)
        return stitch_plan

    def transform_output(self, video_path, context: EffectContext):
        """拼接后作用于最终成片（增强/超分/插帧）。空管线透传。"""
        for m in self.modules:
            try:
                video_path = m.transform_output(video_path, context)
            except Exception as e:
                logger.warning("effect %s.transform_output 失败: %s", getattr(m, "name", "?"), e)
        return video_path
