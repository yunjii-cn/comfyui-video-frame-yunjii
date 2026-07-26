import os
import logging
import subprocess

from .base import EffectModule, EffectContext

logger = logging.getLogger(__name__)


class EnhanceModule(EffectModule):
    """增强模块 —— P3-C。拼接后的成片后处理，与生成链完全解耦。

    通过 ffmpeg 做两件最快见效的事：
    1. 超分：scale 2x（lanczos），常用 480p → 960p；
    2. 帧率提升：minterpolate 运动补偿插帧（SCAIL-2 / WanVideo 输出多为 16fps，
       插到 32fps 对「完美模仿」观感提升最大）。

    关于 RIFE：RIFE 质量更高，但它是 ComfyUI 节点（ComfyUI-Frame-Interpolation）
    而非 CLI 工具，无法在"拼接后文件"这一步直接调用。此处以 minterpolate 作
    **依赖-free 的兜底实现**，结构上已预留 —— 后续可把 transform_output 改为
    向 ComfyUI 提交 RIFE 子图任务以获更高质量。

    ffmpeg 缺失 / 执行失败 → 返回原路径并告警（不崩、不丢片，零回归安全网）。
    """

    stage = 20
    name = "enhance"

    def __init__(self, upscale_factor=2, target_fps=32, crf=18,
                 ffmpeg="ffmpeg", timeout=300):
        self.upscale_factor = int(upscale_factor)
        self.target_fps = int(target_fps) if target_fps else None
        self.crf = int(crf)
        self.ffmpeg = ffmpeg
        self.timeout = int(timeout)

    # ---- 命令构造（纯函数，暴露以便无 ffmpeg 环境也能单测）----
    def _upscale_vf(self, factor):
        f = int(factor)
        if f <= 1:
            return None
        # -2 保证宽为偶数（yuv420p 要求），高度按比例
        return f"scale=iw*{f}:-2:flags=lanczos"

    def _interp_vf(self, fps):
        if not fps or fps <= 1:
            return None
        return (f"minterpolate=fps={int(fps)}:mi_mode=mci:"
                f"mc_mode=aobmc:me_mode=bidir:me=epzs:search_param=32")

    def _out_path(self, in_path, tag):
        d, name = os.path.split(in_path)
        base, ext = os.path.splitext(name)
        return os.path.join(d, f"{base}_{tag}{ext or '.mp4'}")

    def _build_cmd(self, in_path, vf, out_path):
        return [
            self.ffmpeg, "-y", "-i", in_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(self.crf),
            "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "copy", out_path,
        ]

    def _run_ffmpeg(self, in_path, vf, tag):
        out_path = self._out_path(in_path, tag)
        cmd = self._build_cmd(in_path, vf, out_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            logger.warning("EnhanceModule: 未找到 ffmpeg('%s')，跳过%s", self.ffmpeg, tag)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("EnhanceModule: ffmpeg %s 超时(%ss)", tag, self.timeout)
            return None
        if result.returncode != 0 or not os.path.isfile(out_path):
            logger.warning("EnhanceModule: ffmpeg %s 失败: %s", tag, (result.stderr or "")[-300:])
            return None
        return out_path

    def transform_output(self, video_path, context: EffectContext):
        if not video_path or not os.path.isfile(video_path):
            return video_path

        cur = video_path
        intermediates = []

        if self.upscale_factor and self.upscale_factor > 1:
            vf = self._upscale_vf(self.upscale_factor)
            if vf:
                out = self._run_ffmpeg(cur, vf, f"x{self.upscale_factor}")
                if out:
                    intermediates.append(cur)
                    cur = out

        if self.target_fps:
            vf = self._interp_vf(self.target_fps)
            if vf:
                out = self._run_ffmpeg(cur, vf, f"{self.target_fps}fps")
                if out:
                    intermediates.append(cur)
                    cur = out

        # 清理本模块生成的中间文件（绝不删原始输入）
        for mid in intermediates:
            if mid != video_path and mid != cur and os.path.isfile(mid):
                try:
                    os.remove(mid)
                except OSError:
                    pass

        return cur
