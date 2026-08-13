"""稳健的视频首帧封面抽取工具。

为什么需要它：
ComfyUI 画廊（底部 Queue Gallery）只消费节点输出里的 `images` 字段，
根本不读 `gifs`/`videos`（已核对秋叶 HeiHe fork 前端：全量 JS 无 `gifs` 关键字，
媒体 URL 系统 getNodeImageUrls/buildImageUrls 全部基于 `images`）。

节点自身的视频播放器来自 `videos`/`gifs`，而画廊来自 `images`——这是两条不同的
消费路径。因此我们自定义节点要在画廊里显示，必须像标准 SaveImage/PreviewImage
那样往 `images` 写一张可加载的封面 PNG。

原先 _build_output_ui 用 cv2.VideoCapture 抽首帧，但拼接后的成片由 ffmpeg 编码，
OpenCV 的解码后端可能抽不出帧（浏览器却能用 /view 正常播放），导致 `images` 为空、
画廊全白。这里改用 ffmpeg 抽帧（更稳的解码器），cv2 仅作兜底。
"""

import os
import shutil
import subprocess


def extract_poster_png(video_path, out_png=None, timeout=60):
    """从视频抽取首帧 PNG 作为画廊封面。成功返回 png 路径，失败返回 ''。

    - 优先 ffmpeg（本机已依赖，解码最稳，浏览器能播的它基本都能抽）。
    - cv2 兜底。
    """
    if not video_path or not os.path.isfile(video_path):
        return ""

    if out_png is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        parent = os.path.dirname(video_path)
        out_png = os.path.join(parent, stem + "_first.png")

    # 1) ffmpeg 优先：抽首帧为 PNG
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", video_path, "-frames:v", "1", out_png],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            if os.path.isfile(out_png) and os.path.getsize(out_png) > 0:
                return out_png
        except Exception:
            pass

    # 2) cv2 兜底
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            cv2.imwrite(out_png, frame)
            if os.path.isfile(out_png) and os.path.getsize(out_png) > 0:
                return out_png
    except Exception:
        pass

    return ""


def poster_to_image_tensor(png_path, fallback_size=(512, 288)):
    """把封面 PNG 转为 ComfyUI 标准 IMAGE 张量 (1, H, W, 3) float32 ∈ [0,1]。

    作为 YunjiiSegmentStitcher / YunjiiVideoImitator 的 IMAGE 输出（封面帧），
    使本节点成为一等公民的「标准输出节点」，画廊与下游 IMAGE 消费节点都能直接吃。
    失败(无图/缺库)返回 1×1 黑图张量，保证节点执行不崩。torch/numpy/PIL 均 lazy import，
    不增加模块加载成本，且沙箱缺库时不影响 import。
    """
    try:
        import numpy as np
        import torch
        from PIL import Image
        if png_path and os.path.isfile(png_path):
            img = Image.open(png_path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr)[None,]
    except Exception:
        pass
    # 兜底：1×1 黑图（避免节点因封面缺失而报错）
    import numpy as np
    import torch
    return torch.zeros((1, fallback_size[1], fallback_size[0], 3), dtype=torch.float32)
