import os
import logging
import subprocess

from .base import EffectModule, EffectContext
from ..types import STITCH_AUTO, STITCH_CROSS_DISSOLVE, STITCH_HARD_CUT

logger = logging.getLogger(__name__)


# 运镜预设：注入到每段 prompt 的相机运动描述（与 mimic 的身份后缀互补叠加）。
# 所有值均为英文——底层 WanVideo/SCAIL-2 的 prompt 以英文为主，中文会被按 token 当噪声。
CAMERA_MOTION_PRESETS = {
    "电影感": "cinematic camera work, subtle slow dolly, filmic composition",
    "推拉": "slow dolly in and out, dynamic perspective shift",
    "环绕": "slow orbital camera movement around the subject",
    "手持": "natural handheld camera, subtle breathing motion",
    "固定": "static locked-off camera, no camera movement",
    "升降": "smooth crane shot, gentle vertical camera rise",
    "横移": "smooth horizontal tracking pan",
    "俯仰": "gentle tilt up and down, composed framing",
    "跟随": "tracking shot closely following the subject",
}

# 转场预设：映射为 stitcher 已有的 (mode, fade_frames)。
# stitcher 仅原生支持 hard_cut / cross_dissolve 两种，故这里用「不同淡化帧数」表达转场强度，
# 更高级的转场（闪白/光圈/像素化）需 ffmpeg xfade，留待后续 transform_output 扩展。
TRANSITION_PRESETS = {
    "硬切": (STITCH_HARD_CUT, 0),
    "叠化": (STITCH_CROSS_DISSOLVE, 15),
    "电影叠化": (STITCH_CROSS_DISSOLVE, 20),
    "快叠": (STITCH_CROSS_DISSOLVE, 8),
    "长叠": (STITCH_CROSS_DISSOLVE, 30),
    "柔光叠化": (STITCH_CROSS_DISSOLVE, 12),
    "戏剧叠化": (STITCH_CROSS_DISSOLVE, 25),
    "瞬切": (STITCH_HARD_CUT, 0),
    "黑场过渡": (STITCH_CROSS_DISSOLVE, 18),
    "轻叠化": (STITCH_CROSS_DISSOLVE, 10),
}

# 电影调色预设：ffmpeg libavfilter 标准滤镜链（缺滤镜/失败则跳过，返回原片）。
# 仅用 eq / colorbalance / vignette / hue 这些几乎所有 ffmpeg 构建都带的基础滤镜，降低环境依赖风险。
CINEMATIC_GRADES = {
    "none": None,
    "电影感": ("eq=contrast=1.12:saturation=1.08:brightness=-0.01,"
               "colorbalance=rs=0.03:gs=0.0:bs=-0.03,"
               "vignette=PI/4"),
    "清新": ("eq=contrast=1.04:saturation=1.12,"
             "colorbalance=rs=-0.02:gs=0.02:bs=0.02"),
    "复古": ("eq=contrast=1.08:saturation=0.85:brightness=0.01,"
             "colorbalance=rs=0.05:gs=0.0:bs=-0.05,"
             "vignette=PI/5"),
    "黑白": "hue=s=0,eq=contrast=1.15",
    "高对比": "eq=contrast=1.25:saturation=1.05",
}


class CinematicModule(EffectModule):
    """运镜 / 电影感模块 —— P3-D。

    把 PLAN 里的 CAMERA_MOTION_PRESETS / TRANSITION_PRESETS 落地为真实、可消费的效果：
    - 运镜预设 → transform_prompts：给每段 prompt 追加相机运动描述（与 mimic 的身份后缀叠加，
      且带「已存在则不重复追加」保护，可反复跑不叠加）；
    - 转场预设 → transform_stitch：映射为 (mode, fade_frames)，扩展 stitcher 的交叉淡化为转场模板库；
    - 电影调色 → transform_output：对最终成片经 ffmpeg 做色彩分级（缺 ffmpeg / 失败返回原片，零回归）。

    默认不含调色（grade="none"），需显式开启；transform_params 默认透传，
    stabilize=True 时仅轻微抬 cfg / 降 denoise 以求连贯构图，不破坏用户显式设定。

    与 mimic 的关系：mimic(stage=10) 先跑（身份后缀 + 抬高淡化帧），cinematic(stage=15) 后跑，
    可在此基础上叠加运镜、并据自身转场预设二次设定淡化帧——两段各司其职、互不覆盖出错。
    """

    stage = 15
    name = "cinematic"

    def __init__(self, preset="电影感", transition="叠化", grade="none",
                 stabilize=False, ffmpeg="ffmpeg", timeout=300):
        self.preset = preset if preset in CAMERA_MOTION_PRESETS else "电影感"
        self.transition = transition if transition in TRANSITION_PRESETS else "叠化"
        self.grade = grade if grade in CINEMATIC_GRADES else "none"
        self.stabilize = bool(stabilize)
        self.ffmpeg = ffmpeg
        self.timeout = int(timeout)

    # ---------- 运镜：transform_prompts ----------
    def transform_prompts(self, prompts, context: EffectContext):
        if not prompts:
            return prompts
        suffix = CAMERA_MOTION_PRESETS.get(self.preset)
        if not suffix:
            return prompts
        out = []
        for p in prompts:
            p = p or ""
            if suffix in p:  # 防重复：已含该运镜描述则不追加
                out.append(p)
                continue
            sep = ", " if p.strip() else ""
            out.append(f"{p}{sep}{suffix}")
        return out

    # ---------- 参数：transform_params（默认透传，stabilize 时轻调）----------
    def transform_params(self, params, context: EffectContext):
        if not self.stabilize or not isinstance(params, dict):
            return params
        p = dict(params)
        # 仅做温和下限保护，避免覆盖用户明确设定（用 max/min 不降反升时安全）
        cur_cfg = p.get("cfg")
        if isinstance(cur_cfg, (int, float)):
            p["cfg"] = max(float(cur_cfg), 6.0)
        cur_denoise = p.get("denoise")
        if isinstance(cur_denoise, (int, float)):
            p["denoise"] = min(float(cur_denoise), 0.7)
        return p

    # ---------- 转场：transform_stitch ----------
    def transform_stitch(self, stitch_plan, context: EffectContext):
        if not stitch_plan:
            return stitch_plan
        mode, fade = TRANSITION_PRESETS.get(self.transition, TRANSITION_PRESETS["叠化"])
        sp = dict(stitch_plan)
        sp["mode"] = mode
        sp["fade_frames"] = fade
        return sp

    # ---------- 调色：transform_output（ffmpeg，安全降级）----------
    def _grade_vf(self):
        return CINEMATIC_GRADES.get(self.grade)

    def _out_path(self, in_path, tag):
        d, name = os.path.split(in_path)
        base, ext = os.path.splitext(name)
        return os.path.join(d, f"{base}_{tag}{ext or '.mp4'}")

    def _run_ffmpeg(self, in_path, vf, out_path):
        cmd = [
            self.ffmpeg, "-y", "-i", in_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "copy", out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            logger.warning("CinematicModule: 未找到 ffmpeg('%s')，跳过调色", self.ffmpeg)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("CinematicModule: ffmpeg 调色超时(%ss)", self.timeout)
            return None
        if result.returncode != 0 or not os.path.isfile(out_path):
            logger.warning("CinematicModule: ffmpeg 调色失败: %s", (result.stderr or "")[-300:])
            return None
        return out_path

    def transform_output(self, video_path, context: EffectContext):
        vf = self._grade_vf()
        if not vf or not video_path or not os.path.isfile(video_path):
            return video_path
        out_path = self._out_path(video_path, "grade")
        graded = self._run_ffmpeg(video_path, vf, out_path)
        # 不删除输入（可能是 enhance 的输出或用户原片）；仅返回调色结果或原片
        return graded if graded else video_path
