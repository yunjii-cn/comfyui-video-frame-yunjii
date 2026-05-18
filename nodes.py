"""
Yunjii Video Preprocessing Plugin
云集智能 - 视频预处理插件
Copyright 2026
"""

import os
import json
import random
import folder_paths
import cv2
import numpy as np
import torch


class MotionAnalysisNode:
    CATEGORY = "Yunjii/Video"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING", "STRING", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("运动提示词", "分段信息", "镜头数", "关键帧", "帧信息")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        video_files = []
        for f in os.listdir(input_dir) if os.path.isdir(input_dir) else []:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_files.append(f)

        return {
            "required": {
                "视频文件": (sorted(video_files) if video_files else ["(无视频文件)"],
                    {"tooltip": "选择 input/ 目录中的视频文件"}),
                "分段模式": (["自然镜头", "均匀时长", "固定段数"], {"default": "自然镜头",
                    "tooltip": "自然镜头=按画面切换分段；均匀时长=按秒数均分；固定段数=指定段数"}),
                "限制最长秒数": ("BOOLEAN", {"default": True,
                    "tooltip": "开启后，超过最大秒数的段会自动拆分"}),
                "每段最大秒数": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 60.0, "step": 1.0}),
                "限制最短秒数": ("BOOLEAN", {"default": True,
                    "tooltip": "开启后，短于最小秒数的段会合并到相邻段"}),
                "每段最小秒数": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5}),
                "固定段数": ("INT", {"default": 5, "min": 1, "max": 50}),
                "灵敏度": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "保存关键帧": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "视频路径": ("STRING", {"default": ""}),
                "合并指令": ("STRING", {"default": "",
                    "tooltip": "如: 1+2 合并第1和第2段, 多条用逗号分隔"}),
                "拆分指令": ("STRING", {"default": "",
                    "tooltip": "如: 3:2 把第3段拆成2段, 多条用逗号分隔"}),
                "帧偏移": ("STRING", {"default": "",
                    "tooltip": "手动微调关键帧位置，逗号分隔，如: 0,-5,10"}),
                "人物质量词": ("STRING", {"default": "eyes open, clear face, sharp focus",
                    "tooltip": "有人镜头自动追加的质量提示词，留空则不追加"}),
                "手动选帧": ("STRING", {"default": "",
                    "tooltip": "手动选择的帧号，逗号分隔，如: 45,120,195。可通过手动选帧按钮设置"}),
            }
        }

    def analyze(self, 视频文件, 分段模式="自然镜头", 限制最长秒数=True, 每段最大秒数=10.0,
                限制最短秒数=True, 每段最小秒数=2.0, 固定段数=5, 灵敏度=0.3,
                保存关键帧=True, 视频路径="", 合并指令="", 拆分指令="", 帧偏移="",
                人物质量词="eyes open, clear face, sharp focus", 手动选帧=""):

        if 视频路径 and os.path.isfile(视频路径):
            video_path = 视频路径
        else:
            video_path = os.path.join(folder_paths.get_input_directory(), 视频文件)

        if not os.path.isfile(video_path):
            return ("", f"⚠ 找不到视频文件: {video_path}", 0, torch.zeros((1, 512, 512, 3)), "")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return ("", "⚠ 无法打开视频", 0, torch.zeros((1, 512, 512, 3)), "")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()

            max_dur_frames = int(每段最大秒数 * fps) if 限制最长秒数 else total_frames
            min_dur_frames = int(每段最小秒数 * fps) if 限制最短秒数 else 0

            scene_boundaries = _detect_scene_boundaries(video_path, sensitivity=灵敏度, sample_step=2)

            if 分段模式 == "自然镜头":
                scenes = _split_natural(total_frames, fps, max_dur_frames, min_dur_frames, scene_boundaries)
                mode_desc = f"自然镜头模式 → {len(scenes)}段"
            elif 分段模式 == "均匀时长":
                scenes = _split_even(total_frames, fps, max_dur_frames, scene_boundaries)
                mode_desc = f"均匀时长模式 → {len(scenes)}段"
            else:
                scenes = _split_fixed(total_frames, fps, 固定段数, scene_boundaries)
                mode_desc = f"固定段数模式 → {len(scenes)}段"

            if 合并指令.strip():
                scenes = _apply_merge(scenes, 合并指令)
            if 拆分指令.strip():
                scenes = _apply_split(scenes, 拆分指令)

            has_person = _detect_person_in_scenes(video_path, scenes)
            motion_prompts = _analyze_motion(video_path, scenes, fps)

            frame_offsets = []
            if 帧偏移.strip():
                for part in 帧偏移.strip().split(","):
                    try:
                        frame_offsets.append(int(part.strip()))
                    except ValueError:
                        frame_offsets.append(0)
                while len(frame_offsets) < len(scenes):
                    frame_offsets.append(0)

            if 人物质量词.strip():
                quality = 人物质量词.strip()
                for i in range(len(motion_prompts)):
                    if i < len(has_person) and has_person[i]:
                        if quality not in motion_prompts[i]:
                            motion_prompts[i] = (motion_prompts[i] + ", " + quality) if motion_prompts[i] else quality

            auto_frame_map = {}
            for i, (start, end, duration) in enumerate(scenes):
                mid_frame = start + duration // 2
                if frame_offsets and i < len(frame_offsets):
                    mid_frame = max(start, min(mid_frame + frame_offsets[i], end - 1))
                auto_frame_map[mid_frame] = i

            manual_frame_set = set()
            if 手动选帧.strip():
                for part in 手动选帧.strip().split(","):
                    try:
                        fn = int(part.strip())
                        if 0 <= fn < total_frames:
                            manual_frame_set.add(fn)
                    except ValueError:
                        pass

            all_frame_numbers = sorted(set(list(auto_frame_map.keys()) | manual_frame_set))

            keyframes_dir = ""
            if 保存关键帧:
                safe_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = folder_paths.get_output_directory()
                keyframes_dir = os.path.join(output_dir, f"keyframes_{safe_name}")
                os.makedirs(keyframes_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            keyframe_images = []
            frame_info_lines = []
            saved_count = 0

            for fidx in all_frame_numbers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
                keyframe_images.append(tensor)

                is_manual = fidx in manual_frame_set
                scene_idx = auto_frame_map.get(fidx, -1)
                person = has_person[scene_idx] if scene_idx >= 0 and scene_idx < len(has_person) else None

                source = "手动" if is_manual and scene_idx < 0 else ("手动+自动" if is_manual else "自动")
                person_tag = "👤" if person is True else ("🌅" if person is False else "")
                scene_tag = f"镜头{scene_idx+1}" if scene_idx >= 0 else ""
                sec = fidx / fps

                info_line = f"帧#{fidx},{sec:.1f}s,{scene_tag},{person_tag},{source}"
                frame_info_lines.append(info_line)

                if 保存关键帧:
                    tag = "有人物" if person is True else ("无人物" if person is False else "手动")
                    filename = f"frame_{fidx:05d}_{tag}.png"
                    cv2.imwrite(os.path.join(keyframes_dir, filename), frame)
                    saved_count += 1

            cap.release()

            lines = []
            total_sec = total_frames / fps
            lines.append(f"📹 视频分析: {total_frames}帧, {fps:.1f}fps, {total_sec:.1f}秒")
            lines.append(f"🎬 {mode_desc}\n")

            if 保存关键帧:
                lines.append(f"💾 已保存 {saved_count} 个关键帧到: {keyframes_dir}\n")

            for i, (start, end, duration) in enumerate(scenes):
                person = has_person[i] if i < len(has_person) else True
                start_sec = start / fps
                end_sec = end / fps
                dur_sec = duration / fps
                tag = "👤" if person else "🌅"
                motion = motion_prompts[i] if i < len(motion_prompts) else ""
                lines.append(f"  镜头{i+1}: {start_sec:.1f}s-{end_sec:.1f}s ({dur_sec:.1f}秒) {tag}")
                lines.append(f"         ✅ 需要参考图 ({'人物+场景' if person else '场景'})")
                if motion:
                    lines.append(f"         🎬 {motion}")

            if manual_frame_set:
                lines.append(f"\n📌 手动选帧: {len(manual_frame_set)}个额外帧")

            lines.append(f"\n📋 共 {len(all_frame_numbers)} 个关键帧 (自动{len(auto_frame_map)}+手动{len(manual_frame_set - set(auto_frame_map.keys()))})")
            lines.append(f"\n📝 提示词 (||| 分隔):")
            lines.append(" ||| ".join(motion_prompts))

            keyframe_tensor = torch.cat(keyframe_images, dim=0) if keyframe_images else torch.zeros((1, 512, 512, 3))
            frame_info = "\n".join(frame_info_lines)

            scene_json = json.dumps([{
                "start": s, "end": e, "duration": d,
                "person": bool(has_person[i] if i < len(has_person) else True),
                "prompt": motion_prompts[i] if i < len(motion_prompts) else ""
            } for i, (s, e, d) in enumerate(scenes)])

            return {"ui": {"scenes": scene_json}, "result": (
                " ||| ".join(motion_prompts),
                "\n".join(lines),
                len(scenes),
                keyframe_tensor,
                frame_info
            )}

        except Exception as e:
            import traceback
            return ("", f"⚠ 分析失败: {e}\n{traceback.format_exc()}", 0, torch.zeros((1, 512, 512, 3)), "")


class PromptControlNode:
    CATEGORY = "Yunjii/Video"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("分段提示词", "提示词详情")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "自动提示词": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "由运动分析节点生成的自动提示词，用 ||| 分隔"}),
                "自定义前缀": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "每段提示词的自定义前缀，用 ||| 分隔"}),
                "自定义后缀": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "每段提示词的自定义后缀，用 ||| 分隔"}),
                "模式": (["合并", "替换", "前缀", "后缀"], {"default": "合并"}),
            },
            "optional": {
                "分段数": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    def process(self, 自动提示词="", 自定义前缀="", 自定义后缀="", 模式="合并", 分段数=0):
        auto_prompts = [p.strip() for p in 自动提示词.split("|||") if p.strip()] if 自动提示词 else []
        prefixes = [p.strip() for p in 自定义前缀.split("|||")] if 自定义前缀 else []
        suffixes = [p.strip() for p in 自定义后缀.split("|||")] if 自定义后缀 else []

        count = 分段数 if 分段数 > 0 else max(len(auto_prompts), len(prefixes), len(suffixes)) or 1

        result = []
        detail_lines = []
        for i in range(count):
            auto = auto_prompts[i] if i < len(auto_prompts) else ""
            prefix = prefixes[i] if i < len(prefixes) else ""
            suffix = suffixes[i] if i < len(suffixes) else ""

            if 模式 == "替换":
                combined = prefix if prefix else auto
            elif 模式 == "前缀":
                parts = [prefix, auto] if prefix else [auto]
                combined = ", ".join([p for p in parts if p])
            elif 模式 == "后缀":
                parts = [auto, suffix] if suffix else [auto]
                combined = ", ".join([p for p in parts if p])
            else:
                parts = [prefix, auto, suffix]
                combined = ", ".join([p for p in parts if p])

            result.append(combined)
            detail_lines.append(f"  段{i+1}: {combined}")

        detail = f"📝 共 {count} 段提示词 (模式: {模式}):\n" + "\n".join(detail_lines)
        return (" ||| ".join(result), detail)


class KeyframePreviewNode:
    CATEGORY = "Yunjii/Video"
    FUNCTION = "preview"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("关键帧", "帧信息")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "关键帧": ("IMAGE",),
            },
            "optional": {
                "帧信息": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    def preview(self, 关键帧, 帧信息=""):
        ui_data = {"frame_info": 帧信息.split("\n") if 帧信息 else []}
        return {"ui": ui_data, "result": (关键帧, 帧信息)}


def _detect_scene_boundaries(video_path, sensitivity=0.3, sample_step=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    threshold = 1.0 - sensitivity * 0.5
    prev_hist = None
    frame_idx = 0
    boundaries = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step != 0 and frame_idx < total - 1:
            frame_idx += 1
            continue
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < threshold:
                boundaries.append(frame_idx)
        prev_hist = hist
        frame_idx += 1
    cap.release()
    return boundaries


def _split_natural(total_frames, fps, max_dur_frames, min_dur_frames, scene_boundaries):
    boundaries = [0] + scene_boundaries + [total_frames]
    raw = [(boundaries[i], boundaries[i+1], boundaries[i+1] - boundaries[i]) for i in range(len(boundaries)-1)]
    merged = list(raw)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            s, e, d = merged[i]
            if d < min_dur_frames and len(merged) > 1:
                if i > 0 and i < len(merged) - 1:
                    if merged[i-1][2] <= merged[i+1][2]:
                        new_merged[-1] = (new_merged[-1][0], e, e - new_merged[-1][0])
                    else:
                        merged[i+1] = (s, merged[i+1][1], merged[i+1][1] - s)
                        new_merged.append(merged[i+1])
                        i += 1
                elif i > 0:
                    new_merged[-1] = (new_merged[-1][0], e, e - new_merged[-1][0])
                elif i < len(merged) - 1:
                    merged[i+1] = (s, merged[i+1][1], merged[i+1][1] - s)
                    new_merged.append(merged[i+1])
                    i += 1
                changed = True
            else:
                new_merged.append((s, e, d))
            i += 1
        merged = new_merged
    result = []
    for s, e, d in merged:
        if d <= max_dur_frames:
            result.append((s, e, d))
        else:
            num_sub = max(2, (d + max_dur_frames - 1) // max_dur_frames)
            sub_dur = d / num_sub
            for j in range(num_sub):
                sub_s = s + int(sub_dur * j)
                sub_e = s + int(sub_dur * (j + 1)) if j < num_sub - 1 else e
                result.append((sub_s, sub_e, sub_e - sub_s))
    return result


def _split_even(total_frames, fps, max_dur_frames, scene_boundaries):
    num_segments = max(1, (total_frames + max_dur_frames - 1) // max_dur_frames)
    ideal_seg_dur = total_frames / num_segments
    split_points = []
    for seg_i in range(1, num_segments):
        ideal_pos = int(ideal_seg_dur * seg_i)
        search_range = int(ideal_seg_dur * 0.35)
        best_point = ideal_pos
        best_dist = search_range + 1
        for bp in scene_boundaries:
            dist = abs(bp - ideal_pos)
            if dist <= search_range and dist < best_dist:
                best_dist = dist
                best_point = bp
        split_points.append(best_point)
    split_points = sorted(set(split_points))
    boundaries = [0] + split_points + [total_frames]
    return [(boundaries[i], boundaries[i+1], boundaries[i+1] - boundaries[i]) for i in range(len(boundaries)-1)]


def _split_fixed(total_frames, fps, num_segments, scene_boundaries):
    if num_segments <= 1:
        return [(0, total_frames, total_frames)]
    ideal_seg_dur = total_frames / num_segments
    split_points = []
    for seg_i in range(1, num_segments):
        ideal_pos = int(ideal_seg_dur * seg_i)
        search_range = int(ideal_seg_dur * 0.4)
        best_point = ideal_pos
        best_dist = search_range + 1
        for bp in scene_boundaries:
            dist = abs(bp - ideal_pos)
            if dist <= search_range and dist < best_dist:
                best_dist = dist
                best_point = bp
        split_points.append(best_point)
    split_points = sorted(set(split_points))
    boundaries = [0] + split_points + [total_frames]
    return [(boundaries[i], boundaries[i+1], boundaries[i+1] - boundaries[i]) for i in range(len(boundaries)-1)]


def _apply_merge(scenes, merge_cmd):
    if not scenes or not merge_cmd.strip():
        return scenes
    merge_groups = []
    for part in merge_cmd.split(","):
        part = part.strip()
        if not part or "+" not in part:
            continue
        indices = []
        for idx_str in part.split("+"):
            idx_str = idx_str.strip()
            if idx_str.isdigit():
                idx = int(idx_str) - 1
                if 0 <= idx < len(scenes):
                    indices.append(idx)
        if len(indices) >= 2:
            merge_groups.append(sorted(set(indices)))
    merged_set = set()
    for group in merge_groups:
        merged_set.update(group)
    keep_original = set(range(len(scenes))) - merged_set
    result_map = {}
    for idx in sorted(keep_original):
        result_map[idx] = [idx]
    for group in merge_groups:
        result_map[min(group)] = sorted(group)
    result = []
    for key in sorted(result_map.keys()):
        group = result_map[key]
        s = scenes[group[0]][0]
        e = scenes[group[-1]][1]
        result.append((s, e, e - s))
    return result


def _apply_split(scenes, split_cmd):
    if not scenes or not split_cmd.strip():
        return scenes
    split_map = {}
    for part in split_cmd.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        seg = part.split(":")
        if len(seg) == 2 and seg[0].strip().isdigit() and seg[1].strip().isdigit():
            idx = int(seg[0].strip()) - 1
            num = int(seg[1].strip())
            if 0 <= idx < len(scenes) and num >= 2:
                split_map[idx] = num
    result = []
    for i, (s, e, d) in enumerate(scenes):
        if i in split_map:
            num_sub = split_map[i]
            sub_dur = d / num_sub
            for j in range(num_sub):
                sub_s = s + int(sub_dur * j)
                sub_e = s + int(sub_dur * (j + 1)) if j < num_sub - 1 else e
                result.append((sub_s, sub_e, sub_e - sub_s))
        else:
            result.append((s, e, d))
    return result


def _detect_person_in_scenes(video_path, scenes):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [True] * len(scenes)
    has_person = []
    for start, end, duration in scenes:
        detected = False
        for fidx in [start + duration // 4, start + duration // 2, start + 3 * duration // 4]:
            if fidx >= end:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0)) / (160 * 90)
            lower = small[45:90, :]
            edges_lower = cv2.Canny(cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY), 50, 150)
            lower_density = float(np.sum(edges_lower > 0)) / (160 * 45)
            if lower_density > 0.05 and edge_density > 0.04:
                detected = True
                break
        has_person.append(detected)
    cap.release()
    return has_person


def _analyze_motion(video_path, scenes, fps=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [""] * len(scenes)
    prompts = []
    for start, end, duration in scenes:
        dur_sec = duration / fps if fps > 0 else duration / 30
        if dur_sec < 1.0:
            prompts.append("static shot")
            continue
        check_points = [start, start + duration // 3, start + 2 * duration // 3, end - 1]
        frames = []
        for fidx in check_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            frames.append(cv2.resize(frame, (160, 90)) if ret else None)

        motion_parts = []
        if frames[0] is not None and frames[1] is not None:
            g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx, fy = np.mean(flow, axis=(0, 1))
            if abs(fx) > 1.5 and abs(fx) > abs(fy) * 1.5:
                motion_parts.append("camera panning " + ("right" if fx > 0 else "left"))
            elif abs(fy) > 1.5 and abs(fy) > abs(fx) * 1.5:
                motion_parts.append("camera tilting " + ("down" if fy > 0 else "up"))
            elif fx > 1.0 and fy > 1.0:
                motion_parts.append("camera moving diagonally")
            elif abs(fx) < 0.3 and abs(fy) < 0.3:
                motion_parts.append("static shot")

        if frames[0] is not None and frames[3] is not None:
            g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            g3 = cv2.cvtColor(frames[3], cv2.COLOR_BGR2GRAY)
            change_ratio = float(np.sum(cv2.absdiff(g0, g3) > 30)) / (160 * 90)
            if change_ratio > 0.5:
                motion_parts.append("major scene change")
            elif change_ratio > 0.25:
                motion_parts.append("significant movement")

        prompts.append(", ".join(motion_parts) if motion_parts else "smooth motion")
    cap.release()
    return prompts


NODE_CLASS_MAPPINGS = {
    "MotionAnalysisNode": MotionAnalysisNode,
    "PromptControlNode": PromptControlNode,
    "KeyframePreviewNode": KeyframePreviewNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MotionAnalysisNode": "运动分析 🔍 (Yunjii)",
    "PromptControlNode": "提示词控制 📝 (Yunjii)",
    "KeyframePreviewNode": "关键帧预览 🖼 (Yunjii)",
}
