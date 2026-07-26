import os
import json
import time
import logging
import cv2
import numpy as np
import folder_paths

from .types import (
    SegmentResult,
    STITCH_HARD_CUT, STITCH_CROSS_DISSOLVE, STITCH_AUTO,
)
from .debug_log import node_start, node_end, node_error, debug, info, warn

logger = logging.getLogger(__name__)


class YunjiiSegmentStitcher:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "stitch"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("最终视频路径", "拼接报告")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "执行结果": ("STRING", {"default": "", "tooltip": "来自链式执行引擎的执行结果JSON"}),
                "拼接模式": (
                    [STITCH_HARD_CUT, STITCH_CROSS_DISSOLVE, STITCH_AUTO],
                    {"default": STITCH_AUTO, "tooltip": "硬切=直接拼接; 交叉淡化=平滑过渡; 自动=根据内容选择"},
                ),
                "淡化帧数": ("INT", {"default": 8, "min": 2, "max": 30, "step": 1,
                    "tooltip": "交叉淡化过渡帧数"}),
                "输出文件名": ("STRING", {"default": "yunjii_v2v", "tooltip": "输出文件名前缀"}),
            },
            "optional": {
                "音频源": ("STRING", {"default": "", "tooltip": "原始参考视频路径，用于提取音频"}),
            },
        }

    def stitch(self, 执行结果, 拼接模式, 淡化帧数, 输出文件名, 音频源=""):
        node_start("Stitcher", 拼接模式=拼接模式, 淡化帧数=淡化帧数, 输出文件名=输出文件名)

        if not 执行结果.strip():
            node_error("Stitcher", "未提供执行结果")
            node_end("Stitcher", "未提供执行结果")
            return ("", "⚠ 未提供执行结果")

        try:
            results_data = json.loads(执行结果)
        except json.JSONDecodeError as e:
            return ("", f"⚠ 解析执行结果失败: {e}")

        run_id = ""
        if isinstance(results_data, dict):
            run_id = results_data.get("run_id", "")
            segments = results_data.get("segments", [])
        else:
            segments = results_data

        videos = []
        for item in segments:
            if isinstance(item, dict) and item.get("status") == "success":
                vp = item.get("video_path", "")
                if vp and os.path.isfile(vp):
                    videos.append(vp)

        if not videos:
            node_end("Stitcher", "没有成功生成的视频片段")
            return ("", "⚠ 没有成功生成的视频片段可拼接")

        if len(videos) == 1:
            output_path = self._copy_to_output(videos[0], 输出文件名, run_id)
            info("Stitcher", "仅1段视频，直接复制: %s", output_path)
            node_end("Stitcher", f"输出: {output_path}")
            return (output_path, f"✅ 仅1段视频，无需拼接\n输出: {output_path}")

        report_lines = []
        report_lines.append(f"🎬 开始拼接 {len(videos)} 个视频片段")
        report_lines.append(f"📋 拼接模式: {拼接模式}")

        try:
            output_path = self._stitch_videos(videos, 拼接模式, 淡化帧数, 输出文件名, report_lines, run_id)
        except Exception as e:
            logger.error("Stitch failed: %s", e)
            return ("", f"⚠ 拼接失败: {e}")

        if output_path and 音频源 and os.path.isfile(音频源):
            try:
                output_path = self._add_audio(output_path, 音频源, 输出文件名, run_id)
                report_lines.append(f"🎵 已添加原始音频")
            except Exception as e:
                report_lines.append(f"⚠ 音频添加失败: {e}")

        report_lines.append(f"\n✅ 最终输出: {output_path}")
        info("Stitcher", "拼接完成: %s", output_path)
        node_end("Stitcher", f"输出: {output_path}")
        return (output_path, "\n".join(report_lines))

    def _stitch_videos(self, video_paths, mode, fade_frames, output_prefix, report, run_id=""):
        output_dir = folder_paths.get_output_directory()
        sub_dir = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
        yunjii_dir = os.path.join(output_dir, "yunjii_v2v", sub_dir)
        os.makedirs(yunjii_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"{output_prefix}_{timestamp}.mp4"
        output_path = os.path.join(yunjii_dir, output_name)

        ref_cap = cv2.VideoCapture(video_paths[0])
        fps = ref_cap.get(cv2.CAP_PROP_FPS) or 16.0
        width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_cap.release()

        if mode == STITCH_AUTO:
            mode = STITCH_HARD_CUT
            if len(video_paths) > 1:
                diff = self._compute_boundary_diff(video_paths[0], video_paths[1])
                if diff < 30:
                    mode = STITCH_CROSS_DISSOLVE
                report.append(f"📊 段间差异: {diff:.1f}, 自动选择: {mode}")

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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in all_frames:
            writer.write(frame)
        writer.release()

        report.append(f"  总帧数: {len(all_frames)}, 时长: {len(all_frames) / fps:.1f}s")
        return output_path

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
