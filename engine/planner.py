import json
import logging
import math
import re

from .types import (
    SegmentPlan, SegmentInfo,
    SEGMENT_MODE_ONE_SHOT, SEGMENT_MODE_SMART_SPLIT, SEGMENT_MODE_SLIDING_WINDOW,
    REF_STRATEGY_USER_IMAGE, REF_STRATEGY_PREV_LAST_FRAME, REF_STRATEGY_AUTO_SELECT,
    BACKEND_WANVIDEO, BACKEND_SCAIL2,
)
from .debug_log import node_start, node_end, node_error, debug, info, warn

logger = logging.getLogger(__name__)


class YunjiiSegmentPlanner:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "plan"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("段落计划", "段数", "计划摘要")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "分段信息": ("STRING", {"default": "", "tooltip": "来自运动分析节点的分段信息"}),
                "运动提示词": ("STRING", {"default": "", "tooltip": "来自运动分析节点的运动提示词"}),
                "生成模式": (
                    [SEGMENT_MODE_ONE_SHOT, SEGMENT_MODE_SMART_SPLIT, SEGMENT_MODE_SLIDING_WINDOW],
                    {"default": SEGMENT_MODE_ONE_SHOT, "tooltip": "一镜到底=零接缝链式; 智能分段=转场编排; 滑动窗口=超长视频"},
                ),
                "每段最大帧数": ("INT", {"default": 81, "min": 9, "max": 257, "step": 4,
                    "tooltip": "4k+1格式: 41/61/81/85/89/121"}),
                "重叠帧数": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "段间重叠区域大小，一镜到底建议8+"}),
                "目标分辨率": (
                    ["832x480", "480x832", "1280x720", "720x1280"],
                    {"default": "832x480"},
                ),
                "目标帧率": ("INT", {"default": 16, "min": 8, "max": 30}),
                "自适应参数": ("BOOLEAN", {"default": True, "tooltip": "根据复杂度自动调整每段步数/CFG"}),
            },
            "optional": {
                "姿态数据": ("STRING", {"default": ""}),
                "负面提示词": ("STRING", {"default": ""}),
                "生成后端": (
                    ["骨骼路线(WanVideo)", "SCAIL-2 路线"],
                    {"default": "骨骼路线(WanVideo)",
                     "tooltip": "SCAIL-2 路线自动改用「每段81帧/重叠5/步进76」的官方分块规则；需与执行节点的后端选择一致"},
                ),
            },
        }

    def plan(self, 分段信息, 运动提示词, 生成模式, 每段最大帧数, 重叠帧数, 目标分辨率,
             目标帧率, 自适应参数, 姿态数据="", 负面提示词="", 生成后端="骨骼路线(WanVideo)"):

        backend = BACKEND_SCAIL2 if 生成后端 == "SCAIL-2 路线" else BACKEND_WANVIDEO
        if backend == BACKEND_SCAIL2:
            # SCAIL-2 官方分块规则：每段 81 帧、段间重叠 5、有效步进 76。
            # 覆盖用户的帧数/重叠设置，避免段边界跳帧/重影。
            if 每段最大帧数 != 81 or 重叠帧数 != 5:
                info("Planner", "SCAIL-2 路线：分段参数已对齐官方规则（81帧/重叠5/步进76），"
                     "忽略用户设置 每段最大帧数=%d 重叠帧数=%d", 每段最大帧数, 重叠帧数)
            每段最大帧数 = 81
            重叠帧数 = 5

        node_start("Planner", 生成模式=生成模式, 每段最大帧数=每段最大帧数, 重叠帧数=重叠帧数,
                   目标分辨率=目标分辨率, 目标帧率=目标帧率, 生成后端=backend)

        if not 分段信息.strip():
            node_error("Planner", "未提供分段信息")
            return ("", 0, "⚠ 未提供分段信息，请先连接运动分析节点")

        try:
            scenes_raw = self._parse_segment_info(分段信息)
            prompts = self._parse_prompts(运动提示词)
        except Exception as e:
            node_error("Planner", f"解析分段信息失败: {e}")
            return ("", 0, f"⚠ 解析分段信息失败: {e}")

        if not scenes_raw:
            node_error("Planner", "未检测到有效分段")
            return ("", 0, "⚠ 未检测到有效分段")

        original_fps = self._extract_original_fps(分段信息)
        scenes = self._convert_to_target_fps(scenes_raw, original_fps, 目标帧率)

        info("Planner", "解析到 %d 个场景, %d 条提示词, 原始fps=%.1f, 目标fps=%d",
             len(scenes), len(prompts), original_fps, 目标帧率)

        width, height = (int(x) for x in 目标分辨率.split("x"))

        segments = []
        for scene_idx, scene in enumerate(scenes):
            start, end, duration = scene
            complexity = self._estimate_complexity(duration, 目标帧率)

            if 自适应参数:
                seg_frames, steps, cfg = self._adaptive_params(complexity, 每段最大帧数)
            else:
                seg_frames, steps, cfg = 每段最大帧数, 30, 6.0

            # SCAIL-2 路线：段长必须固定 81（官方分块），自适应只调 steps/cfg 不调帧数
            if backend == BACKEND_SCAIL2:
                seg_frames = 每段最大帧数

            if duration <= 每段最大帧数:
                seg_frames = 每段最大帧数
                num_sub = 1
            elif duration <= 每段最大帧数 * 1.5:
                seg_frames = 每段最大帧数
                num_sub = 1
            else:
                num_sub = max(2, math.ceil(duration / seg_frames))

            for sub_idx in range(num_sub):
                sub_start = start + sub_idx * (seg_frames - 重叠帧数)
                sub_end = min(sub_start + seg_frames, end)
                actual_frames = sub_end - sub_start + 1

                actual_frames = max(9, ((actual_frames - 1) // 4) * 4 + 1)

                # 过短的子段（<17帧）Wan 无法独立生成：
                #  - 若有前一段，并入前一段（扩展其 end_frame 并抬高 target_frames）；
                #  - 若是首段（无前一段，仅出现在整段都<17帧时），向前扩展到整段末尾并强制最小 17 帧，
                #    避免无声丢弃视频开头。
                if actual_frames < 17:
                    if segments:
                        segments[-1].end_frame = sub_end
                        segments[-1].target_frames = max(17, ((sub_end - segments[-1].start_frame + 1 - 1) // 4) * 4 + 1)
                        continue
                    # 首段过短：扩展 end_frame 至整段末尾并强制最小帧数（保留片头）
                    sub_end = min(sub_end + (17 - actual_frames), end)
                    actual_frames = max(17, ((sub_end - sub_start + 1 - 1) // 4) * 4 + 1)
                    # 不 continue，落到下方正常建段

                seg_index = len(segments)
                ref_strategy = REF_STRATEGY_USER_IMAGE if seg_index == 0 else (
                    REF_STRATEGY_PREV_LAST_FRAME if 生成模式 == SEGMENT_MODE_ONE_SHOT
                    else REF_STRATEGY_AUTO_SELECT
                )

                prompt = prompts[scene_idx] if scene_idx < len(prompts) else ""
                if 生成模式 == SEGMENT_MODE_ONE_SHOT and seg_index > 0:
                    prompt = prompt or "smooth continuous motion, cinematic"

                overlap_prev = 重叠帧数 if seg_index > 0 else 0
                overlap_next = 重叠帧数 if sub_idx < num_sub - 1 else 0

                segments.append(SegmentInfo(
                    index=seg_index,
                    start_frame=sub_start,
                    end_frame=sub_end,
                    target_frames=actual_frames,
                    overlap_prev=overlap_prev,
                    overlap_next=overlap_next,
                    complexity=complexity,
                    ref_strategy=ref_strategy,
                    prompt=prompt,
                    negative_prompt=负面提示词,
                    params={
                        "steps": steps,
                        "cfg": cfg,
                        "denoise": 1.0,
                        "width": width,
                        "height": height,
                    },
                ))

        plan = SegmentPlan(
            mode=生成模式,
            total_segments=len(segments),
            resolution=[width, height],
            target_fps=目标帧率,
            segments=segments,
            backend=backend,
        )

        plan_json = plan.to_json()
        info("Planner", "规划完成: %d段, 模式=%s, 分辨率=%s", len(segments), 生成模式, 目标分辨率)
        node_end("Planner", f"{len(segments)}段")
        summary = self._build_summary(plan)
        return (plan_json, len(segments), summary)

    def _extract_original_fps(self, info_str):
        fps_match = re.search(r'(\d+(?:\.\d+)?)fps', info_str)
        if fps_match:
            return float(fps_match.group(1))
        return 30.0

    def _convert_to_target_fps(self, scenes_raw, original_fps, target_fps):
        if original_fps <= 0 or target_fps <= 0:
            return scenes_raw

        ratio = target_fps / original_fps
        if abs(ratio - 1.0) < 0.01:
            return scenes_raw

        converted = []
        for start, end, duration in scenes_raw:
            new_start = int(round(start * ratio))
            new_end = int(round(end * ratio))
            new_duration = new_end - new_start
            converted.append((new_start, new_end, new_duration))

        info("Planner", "帧率转换: %.1ffps → %dfps (ratio=%.3f), 帧范围 %d-%d → %d-%d",
             original_fps, target_fps, ratio,
             scenes_raw[0][0], scenes_raw[-1][1],
             converted[0][0], converted[-1][1])

        return converted

    def _parse_segment_info(self, info_str):
        scenes = []
        for line in info_str.strip().split("\n"):
            line = line.strip()
            if not line or "镜头" not in line:
                continue

            m = re.search(r'帧(\d+)-(\d+)', line)
            if m:
                start_frame = int(m.group(1))
                end_frame = int(m.group(2))
                duration = end_frame - start_frame
                scenes.append((start_frame, end_frame, duration))
                continue

            m = re.search(r'(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)s', line)
            if m:
                start_sec = float(m.group(1))
                end_sec = float(m.group(2))
                fps = 30.0
                fps_match = re.search(r'(\d+(?:\.\d+)?)fps', info_str)
                if fps_match:
                    fps = float(fps_match.group(1))
                start_frame = int(start_sec * fps)
                end_frame = int(end_sec * fps)
                duration = end_frame - start_frame
                scenes.append((start_frame, end_frame, duration))
                continue

            parts = line.split()
            for p in parts:
                if p.startswith("帧") and "-" in p:
                    try:
                        range_part = p.split(":")[-1] if ":" in p else p[1:]
                        s, e = range_part.split("-")
                        start_frame = int(s)
                        end_frame = int(e)
                        duration = end_frame - start_frame
                        scenes.append((start_frame, end_frame, duration))
                    except (ValueError, IndexError):
                        pass

        if not scenes:
            try:
                data = json.loads(info_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            s = item.get("start_frame", item.get("start", 0))
                            e = item.get("end_frame", item.get("end", 0))
                            d = e - s
                            scenes.append((s, e, d))
            except (json.JSONDecodeError, TypeError):
                pass

        return scenes

    def _parse_prompts(self, prompts_str):
        prompts = []
        if not prompts_str.strip():
            return prompts
        if "|||" in prompts_str:
            return [p.strip() for p in prompts_str.split("|||") if p.strip()]
        for line in prompts_str.strip().split("\n"):
            line = line.strip()
            if line:
                clean = line
                for prefix in ["镜头1:", "镜头2:", "镜头3:", "镜头4:", "镜头5:",
                               "镜头6:", "镜头7:", "镜头8:", "镜头9:"]:
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):].strip()
                        break
                prompts.append(clean)
        return prompts

    def _estimate_complexity(self, duration_frames, fps):
        duration_sec = duration_frames / max(fps, 1)
        if duration_sec < 2:
            return 0.8
        elif duration_sec < 4:
            return 0.5
        else:
            return 0.3

    def _adaptive_params(self, complexity, max_frames):
        if complexity > 0.7:
            return min(41, max_frames), 30, 6.5
        elif complexity > 0.3:
            return min(61, max_frames), 25, 5.5
        else:
            return max_frames, 20, 5.0

    def _build_summary(self, plan):
        lines = []
        mode_names = {
            SEGMENT_MODE_ONE_SHOT: "一镜到底",
            SEGMENT_MODE_SMART_SPLIT: "智能分段",
            SEGMENT_MODE_SLIDING_WINDOW: "滑动窗口",
        }
        lines.append(f"📋 生成模式: {mode_names.get(plan.mode, plan.mode)}")
        lines.append(f"📊 总段数: {plan.total_segments}")
        lines.append(f"📐 分辨率: {plan.resolution[0]}x{plan.resolution[1]}")
        lines.append(f"🎞 目标帧率: {plan.target_fps}fps")
        lines.append("")

        for seg in plan.segments:
            ref_names = {
                REF_STRATEGY_USER_IMAGE: "👤用户参考图",
                REF_STRATEGY_PREV_LAST_FRAME: "🔗前段末帧",
                REF_STRATEGY_AUTO_SELECT: "🤖自动选取",
            }
            ref_name = ref_names.get(seg.ref_strategy, seg.ref_strategy)
            sec_start = seg.start_frame / plan.target_fps
            sec_end = seg.end_frame / plan.target_fps
            lines.append(
                f"  段{seg.index}: 帧{seg.start_frame}-{seg.end_frame} "
                f"({sec_start:.1f}s-{sec_end:.1f}s) "
                f"| {seg.target_frames}帧 "
                f"| 参考:{ref_name} "
                f"| {seg.params.get('steps', '?')}步 "
                f"| CFG={seg.params.get('cfg', '?')}"
            )

        return "\n".join(lines)
