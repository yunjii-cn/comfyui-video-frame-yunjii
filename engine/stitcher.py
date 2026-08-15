import os
import json
import time
import logging
import subprocess
import cv2
import numpy as np
import folder_paths

from .types import (
    SegmentResult,
    STITCH_HARD_CUT, STITCH_CROSS_DISSOLVE, STITCH_AUTO, STITCH_SEAMLESS,
    STITCH_SEAMLESS_BLEND, STITCH_LATENT_BLEND, STITCH_TRANSITION,
    STITCH_LABELS, STITCH_LABEL_TO_VALUE,
)

# ffmpeg xfade 转场：中文显示名 -> ffmpeg transition 名（仅 ffmpeg转场 拼接模式生效）。
# 这些均为 ffmpeg 原生 xfade 支持的 transition；若值非法，_stitch_videos_xfade 会回退到交叉淡化。
# 特殊键：自动=固定 fade（与默认等价）；随机=每次拼接为每个接缝随机抽取一种（同一次拼接内固定随机种子，可复现）。
XFADE_NAME_MAP = {
    "自动": "fade",
    "淡入淡出": "fade",
    "黑场过渡": "fadeblack",
    "白场过渡": "fadewhite",
    "圆形扩散": "circlecrop",
    "矩形擦除": "rectcrop",
    "横向擦除": "horzclose",
    "纵向擦除": "vertclose",
    "左滑": "slideleft",
    "右滑": "slideright",
    "上滑": "slideup",
    "下滑": "slidedown",
    "溶解": "dissolve",
    "像素化": "pixelize",
    "缩放进入": "zoomin",
    "随机": "__random__",
}
# 随机可抽取的转场池（除 自动/随机 自身外的具体效果）
XFADE_RANDOM_POOL = [v for k, v in XFADE_NAME_MAP.items() if v not in ("fade", "__random__")]
from .pipeline import build_pipeline
from .effects.base import EffectContext
from .debug_log import node_start, node_end, node_error, debug, info, warn

logger = logging.getLogger(__name__)


def _build_xfade_filter(durations, xfade, dur):
    """纯函数：根据各段时长构造 ffmpeg xfade 链式 filter_complex。

    对 N 段视频，xfade 需逐对串联：第 i 次 xfade 的 offset = 当前已合并时长 - 转场时长。
    返回 (filter_complex 字符串, 最后一个输出 label)。各段时长必须 > dur，否则抛 ValueError。
    xfade 可为单一字符串（所有接缝同款）或等长 list（每个接缝独立指定）。
    """
    if len(durations) < 2:
        raise ValueError("xfade 至少需要 2 段视频")
    if isinstance(xfade, (list, tuple)):
        if len(xfade) != len(durations) - 1:
            raise ValueError(f"xfade 列表长度 {len(xfade)} 应与接缝数 {len(durations)-1} 一致")
    else:
        xfade = [xfade] * (len(durations) - 1)
    parts = []
    labels = ["0"]
    acc = durations[0]
    for i in range(1, len(durations)):
        offset = acc - dur
        if offset <= 0:
            raise ValueError(f"段时长过短({durations[i-1]:.2f}s)，无法容纳 {dur:.2f}s xfade")
        prev = labels[-1]
        out_label = f"x0{i}"
        parts.append(
            f"[{prev}][{i}]xfade=transition={xfade[i-1]}:duration={dur:.3f}:offset={offset:.3f}[{out_label}]"
        )
        acc = acc + durations[i] - dur
        labels.append(out_label)
    return ";".join(parts), labels[-1]


def _build_output_ui(output_path: str, first_png: str = "") -> dict:
    """构造 ComfyUI 前端预览所需的 ui 字段，1:1 对齐 VHS_VideoCombine 黄金标准。

    返回 {"gifs": [...], "videos": [...], "images": [...]}：
    - gifs / videos：前端在节点上渲染视频播放器（VHS 只返回 gifs；新版 fork 也认 videos）。
    - images：成片首帧封面兜底——即便视频播放器因个别字段缺失不渲染，节点上也能看到成片。
    - preview 含 workflow(首帧图名) 与 fullpath：前端加载视频前先显示封面，/view 可直接定位。
    字段缺失是「前端节点看不到输出视频」的主因，故这里保证 frame_rate 必有值、workflow/fullpath 必填。
    """
    preview = {
        "filename": os.path.basename(output_path),
        "subfolder": "",
        "type": "output",
        "format": "video/h264-mp4",
        "frame_rate": 16.0,  # 失败时兜底数字，避免缺字段导致前端视频组件不渲染
    }
    first_png = ""
    try:
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        parent = os.path.abspath(os.path.dirname(output_path))
        if parent == output_dir or parent.startswith(output_dir + os.sep):
            _sub = os.path.relpath(parent, output_dir)
            # 前端 /view 走 URL 查询参数(?subfolder=...)，必须统一为正斜杠。
            # 成片恒落在 output/yunjii_v2v/{run_id}/ 子目录，Windows 下 os.path.relpath
            # 返回反斜杠，经 URL 编码后前端构造 /view 易解析失败。正斜杠与 ComfyUI
            # get_save_image_path 的 URL 约定一致，消除 404 隐患。
            preview["subfolder"] = _sub.replace(os.sep, "/")
        # 首帧封面：用 ffmpeg 稳健抽取（画廊只认 images，不读 gifs/videos；
        # cv2 抽帧在 ffmpeg 编码的成片上可能失败，导致画廊空白）。cv2 仅作兜底。
        # first_png 可由调用方预计算并传入（避免重复抽帧）；为空时此处现抽。
        if not first_png:
            try:
                from .poster import extract_poster_png
                first_png = extract_poster_png(output_path)
            except Exception:
                first_png = ""
        # 实际 fps 覆盖兜底值
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(output_path)
            _fps = _cap.get(_cv2.CAP_PROP_FPS)
            _cap.release()
            if _fps and _fps > 0:
                preview["frame_rate"] = float(_fps)
        except Exception:
            pass
        preview["fullpath"] = output_path
    except Exception:
        first_png = ""
    # 首帧封面(workflow) + images 兜底，对齐 VHS 黄金标准，最大化前端渲染兼容。
    # images 是画廊(底部 Queue Gallery)唯一消费的字段——标准 SaveImage/PreviewImage
    # 也正是往 images 写一张图，因此本节点填上 images 即等价于标准输出节点。
    images = []
    if first_png and os.path.isfile(first_png):
        _sub = preview.get("subfolder", "")
        preview["workflow"] = os.path.basename(first_png)
        images.append({"filename": os.path.basename(first_png),
                       "subfolder": _sub, "type": "output"})
    return {"gifs": [preview], "videos": [preview], "images": images}


class YunjiiSegmentStitcher:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "stitch"
    # 末尾追加 IMAGE(封面帧)：本节点成为一等公民「标准输出节点」，
    # 画廊(Queue Gallery)经 images 字段显示封面、节点上经 videos 播放成片，
    # 同时额外吐出一个 IMAGE 张量供下游节点消费。IMAGE 置于末尾，旧连线(视频路径/报告)不受影响。
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("最终视频路径", "拼接报告", "封面帧")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _make_cover(output_path):
        """抽取成片首帧并转为 ComfyUI 标准 IMAGE 张量（封面帧）。返回 (first_png, cover_tensor)。"""
        from .poster import extract_poster_png, poster_to_image_tensor
        first_png = extract_poster_png(output_path)
        return first_png, poster_to_image_tensor(first_png)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "执行结果": ("STRING", {"default": "", "tooltip": "来自链式执行引擎的执行结果JSON"}),
                "拼接模式": (
                    [label for _, label in STITCH_LABELS],
                    {"default": "自动(跟随方案最优)", "tooltip": "默认「自动」即可：SCAIL-2 路线多段→真·零转场一镜到底；骨骼路线多段→交叉淡化平滑过渡。无需再选第二次。其余为手动覆盖：无缝一镜到底(零转场)=硬切去重零转场; 交叉淡化=像素级平滑过渡[转场]; 潜空间交叉淡化=latent 层转场[转场]; 硬切=直接拼接; ffmpeg转场=视频级高级转场(选「转场类型」生效，一镜到底可用作0.5s平滑接缝)"},
                ),
                "淡化帧数": ("INT", {"default": 8, "min": 2, "max": 30, "step": 1,
                    "tooltip": "交叉淡化过渡帧数（仅 交叉淡化 模式生效）"}),
                "转场类型": (list(XFADE_NAME_MAP.keys()), {"default": "淡入淡出",
                    "tooltip": "ffmpeg xfade 视频级转场类型（仅 ffmpeg转场 拼接模式生效）。自动=固定淡入淡出；随机=每个接缝随机抽一种；推荐：淡入淡出"}),
                "转场时长": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "转场时长(秒)，需小于每段时长，否则自动回退交叉淡化。推荐 0.5s"}),
                "输出文件名": ("STRING", {"default": "yunjii_v2v", "tooltip": "输出文件名前缀"}),
            },
            "optional": {
                "音频源": ("STRING", {"default": "", "tooltip": "原始参考视频路径，用于提取音频"}),
                "效果模块": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "可选效果管线模块列表(JSON数组或逗号分隔)，如 [\"mimic\"]。为空=不启用，行为与现状完全一致。支持: mimic, cinematic, enhance, creative（cinematic 可设 xfade 高级转场）"}),
            },
        }

    def stitch(self, 执行结果, 拼接模式, 淡化帧数, 输出文件名, 音频源="", 效果模块="", 转场类型="淡入淡出", 转场时长=0.5):
        # 中文标签 → 英文值归一（兼容旧 saved 英文值 + stitcher 独立调用场景）
        拼接模式 = STITCH_LABEL_TO_VALUE.get(拼接模式, 拼接模式)
        node_start("Stitcher", 拼接模式=拼接模式, 淡化帧数=淡化帧数, 输出文件名=输出文件名)

        if not 执行结果.strip():
            node_error("Stitcher", "未提供执行结果")
            node_end("Stitcher", "未提供执行结果")
            return ("", "⚠ 未提供执行结果", self._make_cover("")[1])

        try:
            results_data = json.loads(执行结果)
        except json.JSONDecodeError as e:
            return ("", f"⚠ 解析执行结果失败: {e}", self._make_cover("")[1])

        run_id = ""
        if isinstance(results_data, dict):
            run_id = results_data.get("run_id", "")
            segments = results_data.get("segments", [])
        else:
            segments = results_data

        videos = []
        # 携带每段重叠帧数(overlap_prev)与 latent 路径，供 seamless / latent_blend 使用
        video_items = []
        for item in segments:
            if isinstance(item, dict) and item.get("status") == "success":
                vp = item.get("video_path", "")
                if vp and os.path.isfile(vp):
                    videos.append(vp)
                    video_items.append({
                        "path": vp,
                        "overlap_prev": int(item.get("overlap_prev", 0) or 0),
                        "segment_index": item.get("segment_index", len(video_items)),
                        "latent_path": item.get("latent_path", "") or "",
                    })

        if not videos:
            node_end("Stitcher", "没有成功生成的视频片段")
            return ("", "⚠ 没有成功生成的视频片段可拼接", self._make_cover("")[1])

        if len(videos) == 1:
            output_path = self._copy_to_output(videos[0], 输出文件名, run_id)
            info("Stitcher", "仅1段视频，直接复制: %s", output_path)
            node_end("Stitcher", f"输出: {output_path}")
            _, cover = self._make_cover(output_path)
            return (output_path, f"✅ 仅1段视频，无需拼接\n输出: {output_path}", cover)

        report_lines = []
        report_lines.append(f"🎬 开始拼接 {len(videos)} 个视频片段")
        report_lines.append(f"📋 拼接模式: {拼接模式}")

        # 效果管线：拼接侧 transform_stitch。空「效果模块」→ 空管线 → 透传，零回归。
        effect_pipeline = build_pipeline(效果模块)
        xfade = ""
        xfade_duration = 0.5
        # ffmpeg转场 模式：直接用 UI 选择的转场类型/时长（优先于效果模块）
        if 拼接模式 == STITCH_TRANSITION:
            _xfade_key = XFADE_NAME_MAP.get(转场类型, "fade")
            if _xfade_key == "__random__":
                # 随机：同一次拼接内用固定随机种子，保证可复现；每个接缝独立抽奖。
                import random as _random
                _rng = _random.Random(12345)
                _n_joints = max(0, len(videos) - 1)
                xfade = [_rng.choice(XFADE_RANDOM_POOL) for _ in range(_n_joints)]
                info("Stitcher", "ffmpeg转场模式: 随机转场, 接缝数=%d", _n_joints)
            else:
                xfade = _xfade_key
            try:
                xfade_duration = float(转场时长)
            except (TypeError, ValueError):
                xfade_duration = 0.5
            info("Stitcher", "ffmpeg转场模式: transition=%s, duration=%.2fs", xfade, xfade_duration)
        if not effect_pipeline.is_empty:
            sp = {"mode": 拼接模式, "fade_frames": 淡化帧数, "add_audio": bool(音频源)}
            sp = effect_pipeline.transform_stitch(sp, EffectContext(metadata={"audio": 音频源}))
            拼接模式 = sp.get("mode", 拼接模式)
            淡化帧数 = int(sp.get("fade_frames", 淡化帧数))
            # 仅当非 ffmpeg转场 模式时，才接受效果模块提供的 xfade
            if not xfade:
                xfade = sp.get("xfade", "") or ""
                try:
                    xfade_duration = float(sp.get("xfade_duration", xfade_duration))
                except (TypeError, ValueError):
                    xfade_duration = 0.5
            info("Stitcher", "已应用效果模块: %s", effect_pipeline.describe())

        # 自动模式：若各段 latent 已落盘（潜空间拼接可用），优先升级为真·一镜到底（潜空间），
        # 过渡最自然；否则沿用像素级无缝近似（_stitch_videos 内部 STITCH_AUTO 逻辑）。
        if 拼接模式 == STITCH_AUTO:
            _latents = [vi.get("latent_path", "") or "" for vi in video_items]
            if _latents and all(_latents):
                info("Stitcher", "自动模式: 各段 latent 可用，升级为真·一镜到底（潜空间拼接）")
                拼接模式 = STITCH_LATENT_BLEND

        try:
            if 拼接模式 == STITCH_LATENT_BLEND:
                # 潜空间拼接：加载各段 latent → 接缝交叉淡化 → 合并解码，过渡最自然。
                # 任意失败(缺 latent/解码异常)自动回退像素重叠混合，保证不丢输出。
                output_path = self._stitch_videos_latent(
                    video_items, 输出文件名, report_lines, run_id)
            else:
                output_path = self._stitch_videos(
                    video_items, 拼接模式, 淡化帧数, 输出文件名, report_lines, run_id,
                    xfade=xfade, xfade_duration=xfade_duration,
                )
        except Exception as e:
            logger.error("Stitch failed: %s", e)
            return ("", f"⚠ 拼接失败: {e}", self._make_cover("")[1])

        if output_path and 音频源 and os.path.isfile(音频源):
            try:
                output_path = self._add_audio(output_path, 音频源, 输出文件名, run_id)
                report_lines.append(f"🎵 已添加原始音频")
            except Exception as e:
                report_lines.append(f"⚠ 音频添加失败: {e}")

        # 效果管线：拼接后 transform_output（增强模块插帧/超分作用于最终成片）。空管线透传。
        if not effect_pipeline.is_empty and output_path and os.path.isfile(output_path):
            enhanced = effect_pipeline.transform_output(output_path, EffectContext(metadata={"audio": 音频源}))
            if enhanced and enhanced != output_path:
                output_path = enhanced
                report_lines.append(f"✨ 已应用增强后处理: {os.path.basename(output_path)}")
                info("Stitcher", "增强后输出: %s", output_path)

        report_lines.append(f"\n✅ 最终输出: {output_path}")
        info("Stitcher", "拼接完成: %s", output_path)
        node_end("Stitcher", f"输出: {output_path}")
        _, cover = self._make_cover(output_path)
        return (output_path, "\n".join(report_lines), cover)

    def _stitch_videos(self, video_items, mode, fade_frames, output_prefix, report, run_id="",
                       xfade="", xfade_duration=0.5):
        # video_items: list of {"path":..., "overlap_prev":int, "segment_index":int}
        video_paths = [v["path"] for v in video_items]

        output_dir = folder_paths.get_output_directory()
        sub_dir = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
        yunjii_dir = os.path.join(output_dir, "yunjii_v2v", sub_dir)
        os.makedirs(yunjii_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"{output_prefix}_{timestamp}.mp4"
        output_path = os.path.join(yunjii_dir, output_name)

        # ffmpeg xfade 高级转场：仅在显式开启且多段时尝试；任何失败都回退下面的旧 cv2 淡化（零回归）。
        if xfade and len(video_paths) > 1:
            try:
                return self._stitch_videos_xfade(
                    video_paths, xfade, xfade_duration, output_path, report, run_id
                )
            except Exception as e:
                logger.warning("Stitcher: xfade 失败，回退普通淡化: %s", e)
                report.append(f"⚠ xfade 转场失败，已回退普通淡化: {e}")
                mode = STITCH_CROSS_DISSOLVE

        ref_cap = cv2.VideoCapture(video_paths[0])
        fps = ref_cap.get(cv2.CAP_PROP_FPS) or 16.0
        width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_cap.release()

        if mode == STITCH_AUTO:
            # 一镜到底等链式模式：段间本就重叠(overlap_prev>0)，再做交叉淡化会出现重影/溶解转场 →
            # 自动改用 seamless_blend 重叠混合（接缝短窗交叉溶解，软化位置跳变但保留零大转场观感）。
            # 非重叠段仍按边界差异决定。
            has_overlap = any(int(v.get("overlap_prev", 0)) > 0 for v in video_items[1:])
            if has_overlap:
                mode = STITCH_SEAMLESS_BLEND
                report.append("📊 检测到段间重叠(一镜到底链式)，自动改为 无缝一镜到底(重叠混合)")
            else:
                mode = STITCH_HARD_CUT
                if len(video_paths) > 1:
                    diff = self._compute_boundary_diff(video_paths[0], video_paths[1])
                    if diff < 30:
                        mode = STITCH_CROSS_DISSOLVE
                    report.append(f"📊 段间差异: {diff:.1f}, 自动选择: {mode}")

        # —— 真·一镜到底：硬切 + 丢弃后续段头部重叠帧 ——
        if mode == STITCH_SEAMLESS:
            all_frames = []
            for i, v in enumerate(video_items):
                frames = self._read_all_frames(v["path"], width, height)
                report.append(f"  段{i}: {len(frames)}帧, {os.path.basename(v['path'])}")
                if i == 0:
                    all_frames.extend(frames)
                else:
                    drop = min(int(v.get("overlap_prev", 0)), max(0, len(frames) - 1))
                    if drop > 0:
                        all_frames.extend(frames[drop:])
                        report.append(f"  ↳ 去重重叠 {drop} 帧（一镜到底）")
                    else:
                        all_frames.extend(frames)
            report.append(f"  总帧数: {len(all_frames)}, 时长: {len(all_frames) / fps:.1f}s")
            return self._write_frames(all_frames, output_path, fps, width, height)

        # —— 一镜到底(重叠混合)：接缝处短窗交叉溶解，软化位置跳变 ——
        # 段i保留完整；段i+1 丢弃头部重叠帧后，用「前段尾 b 帧」与「本段新头 b 帧」做交叉溶解，
        # 把生硬硬切变成几帧的平滑过渡（b=min(重叠帧数, 淡化帧数)）。仅缓解位置不连续；
        # 动作(速度)跳变属生成侧问题，需 previous_frames 时序上下文才能根治（见项目记忆）。
        if mode == STITCH_SEAMLESS_BLEND:
            all_frames = []
            for i, v in enumerate(video_items):
                frames = self._read_all_frames(v["path"], width, height)
                report.append(f"  段{i}: {len(frames)}帧, {os.path.basename(v['path'])}")
                if i == 0:
                    all_frames.extend(frames)
                    continue
                O = int(v.get("overlap_prev", 0))
                if O <= 0 or len(frames) <= O:
                    all_frames.extend(frames)
                    continue
                b = max(2, min(O, int(fade_frames)))
                prev_tail = all_frames[-b:] if len(all_frames) >= b else list(all_frames)
                curr_new = frames[O:O + b]
                n = min(len(prev_tail), len(curr_new))
                blended = []
                for j in range(n):
                    alpha = (j + 1) / b
                    blended.append(cv2.addWeighted(prev_tail[j], 1.0 - alpha, curr_new[j], alpha, 0))
                if len(all_frames) >= b:
                    all_frames[-b:] = blended
                else:
                    all_frames = blended
                all_frames.extend(frames[O + b:])
                report.append(f"  ↳ 接缝混合 {b} 帧（重叠{O}帧, 去重后交叉溶解）")
            report.append(f"  总帧数: {len(all_frames)}, 时长: {len(all_frames) / fps:.1f}s")
            return self._write_frames(all_frames, output_path, fps, width, height)

        all_frames = []

        for i, vp in enumerate(video_paths):
            frames = self._read_all_frames(vp, width, height)
            report.append(f"  段{i}: {len(frames)}帧, {os.path.basename(vp)}")

            if i == 0:
                all_frames.extend(frames)
            else:
                if mode == STITCH_CROSS_DISSOLVE and fade_frames > 0:
                    overlap_prev = all_frames[-fade_frames:]
                    overlap_curr = frames[:fade_frames]

                    for j in range(min(fade_frames, len(overlap_prev), len(overlap_curr))):
                        alpha = (j + 1) / fade_frames
                        blended = cv2.addWeighted(overlap_prev[j], 1.0 - alpha, overlap_curr[j], alpha, 0)
                        all_frames[-fade_frames + j] = blended

                    all_frames.extend(frames[fade_frames:])
                else:
                    all_frames.extend(frames)

        report.append(f"  总帧数: {len(all_frames)}, 时长: {len(all_frames) / fps:.1f}s")
        return self._write_frames(all_frames, output_path, fps, width, height)

    def _write_frames(self, frames, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return output_path

    def _stitch_videos_xfade(self, video_paths, xfade, dur, output_path, report, run_id=""):
        """ffmpeg xfade 真交叉溶解路径：逐段串联（比 cv2 帧级混合更稳、支持光圈/擦除等高级转场）。

        各段需同分辨率/帧率（本管线生成段天然一致）；转场时长 dur 必须 < 每段时长，否则抛错由调用方回退。
        """
        durations = []
        for vp in video_paths:
            cap = cv2.VideoCapture(vp)
            fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            d = (n / fps) if (fps and n > 0) else 0.0
            if d <= dur:
                raise ValueError(f"段时长 {d:.2f}s ≤ 转场时长 {dur:.2f}s，无法用 xfade")
            durations.append(d)

        filter_complex, last_label = _build_xfade_filter(durations, xfade, dur)
        inputs = []
        for vp in video_paths:
            inputs += ["-i", vp]
        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-filter_complex", filter_complex, "-map", f"[{last_label}]",
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", output_path]
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except FileNotFoundError:
            raise RuntimeError("未找到 ffmpeg，无法执行 xfade 转场")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg xfade 超时")
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise RuntimeError("ffmpeg xfade 失败: " + (result.stderr or "")[-400:])
        if isinstance(xfade, (list, tuple)):
            report.append(f"✨ 已用 ffmpeg xfade 随机转场（{len(xfade)} 个接缝，{dur:.2f}s）")
        else:
            report.append(f"✨ 已用 ffmpeg xfade 转场（{xfade}，{dur:.2f}s）")
        return output_path

    def _stitch_videos_latent(self, video_items, output_prefix, report, run_id=""):
        """潜空间拼接（STITCH_LATENT_BLEND）：

        1) 加载各段 latent（YunjiiSaveLatent 落盘的 .pt，含 samples 等全部键）；
        2) 段间接缝处做 latent 空间线性交叉淡化（VAE 时间压缩≈4x，按 overlap_prev 映射 latent 重叠窗），
           生成连贯中间 latent（VAE 解码即连贯中间帧，优于像素叠化）；
        3) 合并 latent 落盘，经 ComfyUI 解码子工作流（原装 WanVideoDecode+VAE）解码为视频；
        4) 任意环节失败 → 回退像素重叠混合（_stitch_videos STITCH_SEAMLESS_BLEND），保证不丢输出。
        """
        import torch
        # 收集有效 latent
        valid = [v for v in video_items if v.get("latent_path") and os.path.isfile(v["latent_path"])]
        if len(valid) < 2:
            report.append("⚠ latent 文件不足(<2)，回退像素重叠混合")
            return self._stitch_videos(video_items, STITCH_SEAMLESS_BLEND, 8, output_prefix, report, run_id)

        try:
            loaded = [torch.load(v["latent_path"], map_location="cpu", weights_only=False) for v in valid]
            tensors = [d["samples"] for d in loaded]
            if any(not isinstance(t, torch.Tensor) or t.dim() != 5 for t in tensors):
                raise ValueError("latent samples 不是 [1,C,T,H,W] 张量")
            # 段间必须真实重叠(overlap_prev>0)潜空间交叉淡化才有意义；多段独立去噪时
            # overlap_prev=0 → 任一接缝 b=0 → torch.cat 纯硬切。此时潜空间混合无意义，
            # 整体回退像素交叉溶解(用 8 帧淡化)，保证不出现硬切跳变。
            _all_overlap = all(
                int(valid[i].get("overlap_prev", 0) or 0) > 0
                for i in range(1, len(valid))
            )
            if not _all_overlap:
                report.append("⚠ 段间无真实重叠(独立生成)，潜空间混合退化为硬切 → 回退像素交叉溶解")
                return self._stitch_videos(video_items, STITCH_CROSS_DISSOLVE, 8, output_prefix, report, run_id)
            # 逐段交叉淡化拼接（累加式）
            merged = tensors[0]
            for i in range(1, len(tensors)):
                cur = tensors[i]
                O = int(valid[i].get("overlap_prev", 0) or 0)
                T_prev, T_cur = merged.shape[2], cur.shape[2]
                b = (O + 2) // 4 if O > 0 else 0   # VAE 时间压缩≈4x：O像素重叠帧 → (O+2)//4 latent帧
                b = min(b, T_prev - 1, T_cur - 1)
                if b <= 0:
                    merged = torch.cat([merged, cur], dim=2)
                    report.append(f"  段{valid[i]['segment_index']}: 无重叠，直接拼接")
                    continue
                prev_tail = merged[:, :, -b:, :, :]
                cur_head = cur[:, :, :b, :, :]
                # 余弦 smoothstep 缓动（两端导数为0），比线性更无痕地把接缝"化开"
                t = torch.linspace(0.0, 1.0, b).view(1, 1, b, 1, 1)
                w = 0.5 - 0.5 * torch.cos(torch.pi * t)
                blended = (1.0 - w) * prev_tail + w * cur_head
                merged = torch.cat([merged[:, :, :-b, :, :], blended, cur[:, :, b:, :, :]], dim=2)
                report.append(f"  段{valid[i]['segment_index']}: latent 重叠淡化 {b} 帧(像素重叠{O})")
            report.append(f"  合并 latent: T={merged.shape[2]}, 形状={list(merged.shape)}")
        except Exception as e:
            logger.warning("Stitcher latent 合并失败，回退像素: %s", e)
            report.append(f"⚠ latent 合并失败({e})，回退像素重叠混合")
            return self._stitch_videos(video_items, STITCH_SEAMLESS_BLEND, 8, output_prefix, report, run_id)

        # 落盘合并 latent 并解码
        output_dir = folder_paths.get_output_directory()
        sub_dir = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
        yunjii_dir = os.path.join(output_dir, "yunjii_v2v", sub_dir)
        os.makedirs(yunjii_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        merged_path = os.path.join(yunjii_dir, f"{output_prefix}_{timestamp}_merged.pt")
        try:
            torch.save({"samples": merged.cpu().contiguous()}, merged_path)
        except Exception as e:
            logger.warning("Stitcher 合并 latent 落盘失败: %s", e)
            report.append(f"⚠ 合并 latent 落盘失败，回退像素重叠混合")
            return self._stitch_videos(video_items, STITCH_SEAMLESS_BLEND, 8, output_prefix, report, run_id)

        decoded = self._decode_latent_via_comfyui(merged_path, run_id, output_prefix, report)
        if decoded and os.path.isfile(decoded):
            report.append(f"✨ 潜空间拼接解码完成: {os.path.basename(decoded)}")
            return decoded
        report.append("⚠ 潜空间解码失败，回退像素重叠混合")
        return self._stitch_videos(video_items, STITCH_SEAMLESS_BLEND, 8, output_prefix, report, run_id)

    def _decode_latent_via_comfyui(self, merged_path, run_id, output_prefix, report):
        """把合并 latent 经 ComfyUI 解码子工作流解码为视频（复用生成时的原装 WanVideoDecode+VAE）。

        解码模板 decode_template.json 由适配器在生成时从真实工作流抽取（VAE 加载器+Decode+VHS 节点
        及参数全保留），本方法仅插入 YunjiiLoadLatent 并把 Decode.samples 重连到它。失败返回 None。
        """
        try:
            output_dir = folder_paths.get_output_directory()
            tmpl_path = os.path.join(output_dir, "yunjii_v2v", run_id, "decode_template.json")
            if not os.path.isfile(tmpl_path):
                logger.warning("Stitcher 缺少 decode 模板: %s", tmpl_path)
                return None
            with open(tmpl_path, encoding="utf-8") as f:
                tmpl = json.load(f)
            nodes = {str(k): v for k, v in tmpl.get("nodes", {}).items()}
            decode_id = str(tmpl.get("decode_id", ""))
            if decode_id not in nodes:
                return None
            # 插入 YunjiiLoadLatent
            nums = [int(k) for k in nodes if str(k).lstrip("-").isdigit()]
            load_id = str((max(nums) + 1) if nums else 9001)
            nodes[load_id] = {"class_type": "YunjiiLoadLatent",
                              "inputs": {"load_path": merged_path}}
            # 重连 Decode.samples -> LoadLatent
            nodes[decode_id].setdefault("inputs", {})["samples"] = [load_id, 0]
            # 重连 VHS.images -> Decode 输出：抽取模板里 VHS 的 images 可能记了被丢弃的
            # 中间节点 id（如 323），导致 POST 时引用幽灵节点、ComfyUI 校验 400。统一改为直连 Decode。
            vhs_id = str(tmpl.get("vhs_id", ""))
            if vhs_id in nodes:
                nodes[vhs_id].setdefault("inputs", {})["images"] = [decode_id, 0]
            wf = {"prompt": nodes}

            import urllib.request
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(
                "http://127.0.0.1:8188/prompt",
                data=json.dumps(wf).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with op.open(req, timeout=30) as r:
                pid = json.loads(r.read()).get("prompt_id")
            if not pid:
                return None
            # 轮询 history
            for _ in range(600):
                time.sleep(2)
                try:
                    with op.open(f"http://127.0.0.1:8188/history/{pid}", timeout=10) as h:
                        hj = json.loads(h.read())
                    if pid in hj:
                        outputs = hj[pid].get("outputs", {})
                        for node_out in outputs.values():
                            for g in node_out.get("gifs", []):
                                fn = g.get("filename", "")
                                sub = g.get("subfolder", "")
                                if fn:
                                    return os.path.join(output_dir, sub, fn) if sub else os.path.join(output_dir, fn)
                        return None
                except Exception:
                    continue
            return None
        except Exception as e:
            logger.warning("Stitcher 潜空间解码异常: %s", e)
            return None

    def _read_all_frames(self, video_path, target_w, target_h):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if w != target_w or h != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames

    def _compute_boundary_diff(self, video1, video2, sample_count=3):
        cap1 = cv2.VideoCapture(video1)
        cap2 = cv2.VideoCapture(video2)
        total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

        last_frames = []
        for idx in range(max(0, total1 - sample_count), total1):
            cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap1.read()
            if ret:
                last_frames.append(frame)

        first_frames = []
        for idx in range(min(sample_count, total2)):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap2.read()
            if ret:
                first_frames.append(frame)

        cap1.release()
        cap2.release()

        if not last_frames or not first_frames:
            return 100.0

        last_gray = cv2.cvtColor(last_frames[-1], cv2.COLOR_BGR2GRAY)
        first_gray = cv2.cvtColor(first_frames[0], cv2.COLOR_BGR2GRAY)

        if last_gray.shape != first_gray.shape:
            first_gray = cv2.resize(first_gray, (last_gray.shape[1], last_gray.shape[0]))

        diff = np.mean(np.abs(last_gray.astype(float) - first_gray.astype(float)))
        return diff

    def _add_audio(self, video_path, audio_source, output_prefix, run_id=""):
        try:
            import subprocess
            output_dir = os.path.dirname(video_path)
            temp_output = os.path.join(output_dir, f"{output_prefix}_with_audio.mp4")

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_source,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                temp_output,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and os.path.isfile(temp_output):
                os.replace(temp_output, video_path)
                return video_path
            else:
                logger.warning("ffmpeg audio add failed: %s", result.stderr[:200])
                return video_path
        except FileNotFoundError:
            logger.warning("ffmpeg not found, skipping audio transfer")
            return video_path
        except Exception as e:
            logger.warning("Audio transfer failed: %s", e)
            return video_path

    def _copy_to_output(self, src_path, output_prefix, run_id=""):
        import shutil
        output_dir = folder_paths.get_output_directory()
        sub_dir = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
        yunjii_dir = os.path.join(output_dir, "yunjii_v2v", sub_dir)
        os.makedirs(yunjii_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(src_path)[1] or ".mp4"
        output_name = f"{output_prefix}_{timestamp}{ext}"
        dst = os.path.join(yunjii_dir, output_name)
        shutil.copy2(src_path, dst)
        return dst


# 模块级别名：composer.py 以 `from .stitcher import _make_cover` 方式引用，
# 而 _make_cover 定义在 YunjiiSegmentStitcher 内为 @staticmethod。此处暴露为模块级，
# 同时类内 `self._make_cover(...)` 调用（staticmethod 经实例访问）依旧有效。
_make_cover = YunjiiSegmentStitcher._make_cover
