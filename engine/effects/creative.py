import importlib.util
import logging
import sys

from .base import EffectModule, EffectContext
from ..types import STITCH_HARD_CUT, STITCH_CROSS_DISSOLVE, STITCH_SEAMLESS, STITCH_SEAMLESS_BLEND

logger = logging.getLogger(__name__)


def _detect_librosa() -> bool:
    """仅探测、不导入：判断 librosa 是否可 import。

    与 ``import librosa`` 语义一致——已在 sys.modules 中、或 find_spec 能解析都算可用。
    刻意不顶层 import librosa——它拉 numpy/scipy/numba 等大依赖，
    未安装时本模块照常可导入/可组合；装好后重启 ComfyUI 即生效。
    """
    try:
        if sys.modules.get("librosa") is not None:
            return True
        return importlib.util.find_spec("librosa") is not None
    except Exception:
        return False


def rhythm_reconstruction_available() -> bool:
    """供节点 / 贯穿节点展示：本环境是否已安装 librosa（音频驱动节奏重建可用）。

    每次调用都重新探测，装好 librosa 并重启后即为 True（无需改代码）。
    启用命令（在 ComfyUI 的 Python 环境里）：pip install librosa
    """
    return _detect_librosa()


# 创意节奏预设：映射为 stitcher 的 (mode, fade_frames)。
# 与 cinematic 转场预设的区别：这里代表「整体剪辑节奏」，由 creative(stage=30) 在 cinematic(15) 之后
# 二次裁决，作为最终节奏。无音频时也能单独使用。
PACING_PRESETS = {
    "顺滑": (STITCH_CROSS_DISSOLVE, 18),
    "动感": (STITCH_HARD_CUT, 0),
    "蒙太奇": (STITCH_CROSS_DISSOLVE, 8),
    "电影感": (STITCH_CROSS_DISSOLVE, 22),
    "呼吸": (STITCH_CROSS_DISSOLVE, 14),
}

# 能量→运动强度描述（英文，注入 prompt；与底层英文 prompt 一致）。
# 把音频 RMS 能量分桶映射为「该段应有的运动强度」，实现「画面运动节奏跟随音乐节奏」。
ENERGY_TIERS = [
    (0.34, "calm, slow and gentle motion"),
    (0.67, "steady, moderate movement"),
    (0.85, "dynamic, energetic motion"),
    (1.01, "explosive, fast-paced motion"),
]


def _tier(norm: float) -> str:
    for thr, desc in ENERGY_TIERS:
        if norm < thr:
            return desc
    return ENERGY_TIERS[-1][1]


class CreativeModule(EffectModule):
    """创意模块 —— P3-E。

    两件事，分层降级：
    1. **节奏重构（音频驱动）**：若提供 ``audio_path`` 且环境装了 ``librosa``，则分析音频 RMS 能量包络，
       按段数均分，给每一段 prompt 注入「该段应有的运动强度」描述，使生成画面运动节奏跟随音乐。
       无 librosa / 无音频 / 分析失败时 → **跳过并告警**，prompt 原样透传（零回归、不崩）。
    2. **创意节奏预设（PACING_PRESETS）**：``transform_stitch`` 把整体剪辑节奏映射为 (mode, fade_frames)，
       在 cinematic 之后二次裁决最终转场；无音频也能用。

    设计原则（呼应 P3 计划「librosa 包体积单独评估」）：
    - **绝不 import librosa 顶层**——只在 ``_analyze_audio`` 内惰性导入，未安装时模块照常可导入、可组合；
    - 真正「时间重映射（按节拍重切段）」需要重跑 planner 用节拍感知的分段边界，属后续增强（M6+），
      本模块在当前架构内做的是「运动强度/节奏预设」层面的节奏重构，已是最优可落地形态。

    依赖评估：librosa 本身几 MB，但会拉 numpy/scipy/numba/soundfile/resampy 等，在已有 torch/numpy 的
    ComfyUI 环境里边际增量约 100–200MB。是否安装由你决定；不装也能用 PACING 预设。

    启用真实节奏重建：在 ComfyUI 的 Python 环境执行 ``pip install librosa`` 并重启 ComfyUI 即可，
    无需改任何代码（``rhythm_reconstruction_available()`` 会变为 True）。模块导入时不触碰 librosa，
    未安装不影响导入与组合。
    """

    stage = 30
    name = "creative"

    def __init__(self, audio_path="", pacing="顺滑", motion_prompt=True, ffmpeg="ffmpeg"):
        self.audio_path = audio_path or ""
        self.pacing = pacing if pacing in PACING_PRESETS else "顺滑"
        self.motion_prompt = bool(motion_prompt)
        self.ffmpeg = ffmpeg  # 预留（librosa 无法读某些封装时借 ffmpeg 抽音频）

    # ---------- 音频能量分析（惰性 librosa，纯 Python 容错）----------
    def _analyze_audio(self, audio_path, num_segments):
        """返回长度=num_segments 的逐段运动描述列表；任何失败返回 None。"""
        if not audio_path or num_segments <= 0:
            return None
        if not _detect_librosa():  # 先探测，避免每次都抛异常刷 warning
            logger.info(
                "CreativeModule: 未安装 librosa，跳过音频节奏分析（仅应用 pacing 预设）。"
                "启用真实节奏重建：在 ComfyUI 的 Python 环境执行 `pip install librosa` 后重启。"
            )
            return None
        try:
            import librosa  # 惰性：未安装则跳到 except
        except Exception:
            logger.warning("CreativeModule: librosa 探测通过但导入失败，跳过音频节奏分析")
            return None
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=600)
            rms = librosa.feature.rms(y=y)
            # 兼容 np.ndarray：转成一维 Python list，避免引擎对 numpy 的硬依赖
            if hasattr(rms, "flatten"):
                env = list(rms.flatten())
            elif hasattr(rms, "__len__"):
                env = list(rms)
            else:
                env = [float(rms)]
            if not env:
                return None
            total = len(env)
            means = []
            for i in range(num_segments):
                a = (i * total) // num_segments
                b = max(a + 1, ((i + 1) * total) // num_segments)
                seg = env[a:b]
                means.append(sum(seg) / len(seg) if seg else 0.0)
            lo, hi = min(means), max(means)
            rng = (hi - lo) or 1.0
            return [_tier((m - lo) / rng) for m in means]
        except Exception as e:
            logger.warning("CreativeModule: 音频分析失败(%s)，跳过节奏重构: %s", audio_path, e)
            return None

    # ---------- 节奏重构：transform_prompts ----------
    def transform_prompts(self, prompts, context: EffectContext):
        if not self.motion_prompt or not prompts:
            return prompts
        descriptors = self._analyze_audio(self.audio_path, len(prompts))
        if not descriptors:
            return prompts  # 无音频/librosa：原样透传
        out = []
        for i, p in enumerate(prompts):
            p = p or ""
            desc = descriptors[i] if i < len(descriptors) else descriptors[-1]
            if desc in p:  # 幂等：已含则不重复追加
                out.append(p)
                continue
            sep = ", " if p.strip() else ""
            out.append(f"{p}{sep}{desc}")
        return out

    # ---------- 创意节奏预设：transform_stitch ----------
    def transform_stitch(self, stitch_plan, context: EffectContext):
        if not stitch_plan:
            return stitch_plan
        sp = dict(stitch_plan)
        # 一镜到底已强制 seamless / seamless_blend：不再注入任何节奏转场（保持零转场连续性）
        if sp.get("mode") in (STITCH_SEAMLESS, STITCH_SEAMLESS_BLEND):
            return sp
        # cinematic(stage=15) 已选 xfade 高级转场时，保留不覆盖（xfade 优先级高于 pacing 的普通淡入淡出）。
        if sp.get("xfade"):
            return sp
        mode, fade = PACING_PRESETS.get(self.pacing, PACING_PRESETS["顺滑"])
        sp["mode"] = mode
        sp["fade_frames"] = fade
        return sp

    # transform_params 默认透传（创意不直接改 steps/cfg，避免与用户设定冲突）
    def transform_params(self, params, context: EffectContext):
        return params
