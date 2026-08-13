import json
import logging
import math
import re

from .types import (
    SegmentPlan, SegmentInfo,
    SEGMENT_MODE_ONE_SHOT, SEGMENT_MODE_SMART_SPLIT, SEGMENT_MODE_SLIDING_WINDOW,
    REF_STRATEGY_USER_IMAGE, REF_STRATEGY_PREV_LAST_FRAME, REF_STRATEGY_AUTO_SELECT,
    BACKEND_WANVIDEO, BACKEND_SCAIL2,
    CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START,
    CONTINUITY_AUTO, CONTINUITY_LABEL_TO_VALUE, CONTINUITY_LABELS,
    SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO,
    SEAMLESS_PLAN_LABELS, SEAMLESS_PLAN_LABEL_TO_VALUE,
)
from .debug_log import node_start, node_end, node_error, debug, info, warn

logger = logging.getLogger(__name__)

# 生成模式兼容别名：早期版本用英文名(one_shot/smart_split/sliding_window)，
# 后因迁移改为中文。这里做兼容，使任何旧工作流(存英文值)也能通过校验并正确运行。
MODE_ALIASES = {
    "one_shot": "一镜到底",
    "smart_split": "智能分段",
    "sliding_window": "滑动窗口",
}


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
                    {"default": SEGMENT_MODE_ONE_SHOT, "tooltip": "一镜到底=零转场连续长镜头(SCAIL-2长视频自动单次超长生成,context滑窗覆盖全帧); 智能分段=转场编排; 滑动窗口=超长视频"},
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
                "单遍连贯模式": ("BOOLEAN", {"default": False,
                    "tooltip": "【已并入连贯策略】旧开关，等价连贯策略=单遍连贯(方案C)。建议改用下方「连贯策略」下拉。"}),
                "连贯策略": (
                    [label for _, label in CONTINUITY_LABELS],
                    {"default": "多段无缝(默认)",
                     "tooltip": "生成侧时序连续性方案：多段无缝(默认,接缝化妆)=分段独立I2V+混合; "
                                "单遍连贯(方案C)=整片一次去噪latent连续真·一镜到底,长视频画质软; "
                                "暖启动(Tier2)=分段+上段真实帧喂回WanAnimatePlus prefix_frames,连续+画质兼得(需SCAIL-2路线+WanAnimatePlus)"},
                ),
                "单遍时长上限": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "方案C单遍最大时长(秒)。>0 时超出则回退多段seamless以抑制长程稀释画质退化；0=不限制(整片单遍)"}),
                "模型精度": (
                    ["fp8", "fp16"],
                    {"default": "fp8", "tooltip": "SCAIL-2 扩散模型精度：fp8(默认,省显存,略软); fp16(更精细,吃显存,需本机有fp16权重或显存充足)"},
                ),
                # —— 注意：新增大下拉务必放 optional 末尾，避免已有工作流 widgets_values 位置错位 ——
                # （历史教训：曾把本下拉插在中段，导致后续 FLOAT 槽 单遍时长上限 收到 模型精度 的 "fp8" 而校验崩溃）
                "无缝连贯方案": (
                    [label for _, label in SEAMLESS_PLAN_LABELS],
                    {"default": "A方案·标准多段无缝(独立生成+平滑过渡, ≤15s) ⭐默认",
                     "tooltip": "【推荐用此下拉替代旧连贯策略来选无缝档位】三档机制各不相同：\n"
                                "· A方案(标准多段无缝)：一般时长≤15s，多段独立生成、接缝交叉溶解平滑过渡；"
                                "每段质量最高、显存友好、可分段重试。段边界为平滑过渡(非真连续)。\n"
                                "· B方案(超长视频无缝)：15~30s+ 长视频，单遍连续采样+context滑窗覆盖全帧"
                                "(81帧一窗/重叠32潜空间fuse)=真·零接缝、长视频无劣化(⭐长片推荐)；"
                                "代价是显存峰值更高、不可分段重试、耗时随总长线性增长。\n"
                                "· C方案(单遍兜底)：整片一次去噪、不注入滑窗，>5s画质软，仅作对比/兜底，不推荐主用。\n"
                                "选B/C时拼接模式可保持默认(单遍无需拼接)；选A时接缝由交叉溶解平滑处理。"},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 兼容旧工作流：早期「生成模式」存英文值(one_shot 等)，节点 COMBO 现仅接受中文。
        # 定义本方法让 ComfyUI 跳过默认 COMBO 成员校验，由 plan() 内部归一化。
        return True

    def plan(self, 分段信息, 运动提示词, 生成模式, 每段最大帧数, 重叠帧数, 目标分辨率,
             目标帧率, 自适应参数, 姿态数据="", 负面提示词="", 生成后端="骨骼路线(WanVideo)",
             单遍连贯模式=False, 连贯策略=CONTINUITY_AUTO, 单遍时长上限=0.0, 模型精度="fp8",
             无缝连贯方案=SEAMLESS_PLAN_AUTO):

        # 归一化：旧工作流可能传英文值(one_shot 等)，统一映射为中文常量
        生成模式 = MODE_ALIASES.get(生成模式, 生成模式)

        backend = BACKEND_SCAIL2 if 生成后端 == "SCAIL-2 路线" else BACKEND_WANVIDEO
        if backend == BACKEND_SCAIL2 and 生成模式 != SEGMENT_MODE_ONE_SHOT:
            # SCAIL-2 分块：每段固定 81 帧（沿用官方）。
            # 段间重叠锁定为 32 像素帧（对齐『三层楼的小肥猴』Wan2.2 Animate 工作流
            #   WanVideoContextOptions 的 context_overlap=32 → 8 latent 帧）；VAE 时间压缩≈4x
            #   → 32/4=8 latent 帧。该值已在猴子工作流验证为自然连贯，故直接采用，
            #   不再地板16（避免用户漏设时混合窗偏窄）。若 UI 设更高值仍尊重(clamp≤32)。
            # 单段生成质量(4步蒸馏)不受影响，仅接缝 latent 交叉淡化窗加宽到 8 帧更柔。
            每段最大帧数 = 81
            _user_ov = 重叠帧数 if isinstance(重叠帧数, int) else 0
            _new_ov = min(max(_user_ov, 32), 32)
            重叠帧数 = _new_ov
            if _user_ov < 32:
                info("Planner", "SCAIL-2 路线：每段固定81帧；段间重叠锁定为32（对齐猴子工作流验证的自然连贯方案，8 latent帧）")

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

        # 方案C 单遍判定：一镜到底 + SCAIL-2 + 长视频(>单段上限×1.5) + 用户显式选单遍/勾选旧开关
        total_frames = (max(e for _, e, _ in scenes) - min(s for s, _, _ in scenes)) if scenes else 0
        total_seconds = total_frames / max(目标帧率, 1)

        # 连贯策略归一：显式选择(非auto)优先；否则回退旧「单遍连贯模式」bool
        _raw_strategy = 连贯策略
        if not _raw_strategy or _raw_strategy == CONTINUITY_AUTO:
            _raw_strategy = CONTINUITY_SINGLE_PASS if 单遍连贯模式 else CONTINUITY_MULTI_SEG
        strategy = CONTINUITY_LABEL_TO_VALUE.get(_raw_strategy, _raw_strategy)
        if strategy not in (CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START):
            strategy = CONTINUITY_MULTI_SEG

        # —— 无缝连贯方案（A/B/C）归一：本下拉是用户选无缝档位的主入口，优先于旧连贯策略 ——
        # 三档机制各不相同：
        #   A → 标准多段无缝(strategy=multi_seg, 每段独立+交叉溶解平滑过渡, 适用≤15s)
        #   B → 超长视频无缝(strategy=single_pass, 单遍连续采样+context滑窗真·无缝, 适用15~30s+)
        #   C → 单遍连贯兜底(strategy=single_pass, 不注入滑窗, >5s画质软, 仅对比/兜底)
        #   auto → 保持上面由连贯策略归一出的 strategy（兼容旧工作流/未选本下拉）
        _seamless = SEAMLESS_PLAN_LABEL_TO_VALUE.get(无缝连贯方案, 无缝连贯方案)
        if _seamless not in (SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO):
            _seamless = SEAMLESS_PLAN_AUTO
        seamless_plan = _seamless
        long_video_mode = False
        if seamless_plan == SEAMLESS_PLAN_A:
            strategy = CONTINUITY_MULTI_SEG
            long_video_mode = False
            info("Planner", "无缝连贯方案=A (标准多段无缝, 一般时长≤15s)")
        elif seamless_plan == SEAMLESS_PLAN_B:
            # B 方案：超长视频无缝 = 单遍连续采样 + context 滑窗覆盖全帧。
            # 整片作为一条去噪轨迹、按 81 帧一窗重叠 32 潜空间 fuse → 真·无漂移、真无缝，
            # 长视频(15~30s+) 无劣化。与 C 同为 single_pass 规划，但 B 注入滑窗(真无缝)、
            # C 不注入(旧兜底, >5s 画质软)。仅一镜到底+SCAIL-2 路线适用滑窗；否则退化多段平滑。
            long_video_mode = True
            if 生成模式 == SEGMENT_MODE_ONE_SHOT and backend == BACKEND_SCAIL2:
                strategy = CONTINUITY_SINGLE_PASS
                info("Planner", "无缝连贯方案=B (超长视频无缝: 单遍连续采样+context滑窗覆盖全帧, 真·无漂移)")
            else:
                strategy = CONTINUITY_MULTI_SEG
                info("Planner", "无缝连贯方案=B 退化为多段无缝(非一镜到底/非SCAIL2, 滑窗不适用)")
        elif seamless_plan == SEAMLESS_PLAN_C:
            strategy = CONTINUITY_SINGLE_PASS
            long_video_mode = False
            info("Planner", "无缝连贯方案=C (单遍连贯·旧方案C兜底)")

        single_pass_requested = (
            strategy == CONTINUITY_SINGLE_PASS
            and 生成模式 == SEGMENT_MODE_ONE_SHOT
            and backend == BACKEND_SCAIL2
            and total_frames > 每段最大帧数 * 1.5
        )
        # 方案C 画质增强：单遍时长上限。超出则回退多段seamless(抑制长程稀释)
        cap = float(单遍时长上限) if 单遍时长上限 else 0.0
        single_pass = single_pass_requested
        # B 方案：一镜到底+SCAIL-2 场景强制单遍连续采样（即便短视频也走连续轨迹，滑窗在窗口内无副作用）；
        # 长视频正是 B 主场(真无缝无劣化)。非适用场景 B 已退化 multi_seg，此处不强制。
        if seamless_plan == SEAMLESS_PLAN_B and strategy == CONTINUITY_SINGLE_PASS:
            single_pass = True
        if single_pass and cap > 0 and total_seconds > cap:
            warn("Planner", "单遍被「单遍时长上限=%.1fs」截断(%d帧≈%.1fs)，回退多段seamless(平滑过渡)抑制画质退化",
                 cap, total_frames, total_seconds)
            single_pass = False
            strategy = CONTINUITY_MULTI_SEG  # 回退为 A 式多段平滑
        if single_pass:
            info("Planner", "单遍连贯模式启用：%s长视频(%d帧) 改为单次超长生成(%s)",
                 "B超长" if seamless_plan == SEAMLESS_PLAN_B else "C",
                 total_frames,
                 "context滑窗真无缝" if seamless_plan != SEAMLESS_PLAN_C else "不滑窗·旧兜底(画质软)")
        elif strategy == CONTINUITY_WARM_START:
            info("Planner", "暖启动(Tier2) 启用：分段 + 上段真实帧喂回 WanAnimatePlus prefix_frames（需SCAIL-2路线+WanAnimatePlus）")

        segments = self._build_segments(
            scenes, backend, 每段最大帧数, 重叠帧数, width, height,
            prompts, 生成模式, 目标帧率, 自适应参数, 负面提示词,
            single_pass=single_pass, long_video_mode=long_video_mode,
        )

        # 一镜到底长视频默认走「多段 seamless」(质量优先)：每段 81 帧(5s 原生)全质量生成，
        # composer 强制 seamless 硬切丢重叠帧 = 零转场、零重复帧，质量对齐好片。
        # 仅当连贯策略=单遍连贯(方案C)时 is_single_pass=True → 整片一次连贯去噪(真连贯，画质代价)。
        # 暖启动(Tier2) 仍为多段，但由适配器注入上段末帧作 prefix，段间连续+画质兼得。
        is_single_pass = single_pass
        plan = SegmentPlan(
            mode=生成模式,
            total_segments=len(segments),
            resolution=[width, height],
            target_fps=目标帧率,
            segments=segments,
            backend=backend,
            single_pass=is_single_pass,
            continuity_strategy=strategy,
            seamless_plan=seamless_plan,
            long_video_mode=long_video_mode,
            model_precision=模型精度 if backend == BACKEND_SCAIL2 else "fp8",
            single_pass_cap=cap,
        )

        plan_json = plan.to_json()
        info("Planner", "规划完成: %d段, 模式=%s, 分辨率=%s", len(segments), 生成模式, 目标分辨率)
        node_end("Planner", f"{len(segments)}段")
        summary = self._build_summary(plan)
        return (plan_json, len(segments), summary)

    def _build_segments(self, scenes, backend, 每段最大帧数, 重叠帧数, width, height,
                        prompts, 生成模式, target_fps, 自适应参数=True, 负面提示词="",
                        single_pass=False, long_video_mode=False):
        """按指定后端的官方分块规则对 scenes 开窗建段。plan() 与 replan_for_backend() 共用。
        long_video_mode(B 方案)：仅做日志与重叠提示，防漂移主体由生成侧 context_options 续写 + 真骨架承担。"""
        # 防御性归一化：兼容旧 plan JSON 里残存的英文模式值(one_shot 等)
        生成模式 = MODE_ALIASES.get(生成模式, 生成模式)
        if long_video_mode:
            info("Planner", "B方案(超长视频无缝): 已启用长程防漂移分块 — 段间重叠=%d 帧(=8 latent), "
                            "每段81帧, 按容量自动分块(81帧×N段, N≈时长/5s, 9段≈30s+); 防漂移由生成侧"
                            "context_options跨段reference_latent续写 + SCAIL真骨架(逐帧姿态)承担", 重叠帧数)
        segments = []

        # —— 方案C 单遍（仅当用户显式勾选「单遍连贯模式」时 single_pass=True）——
        # 一镜到底长视频做成「单段覆盖全长」：整片一次连贯去噪，latent 天然连续 = 真·一镜到底无接缝。
        # 代价：11s 长视频被长程重度混合稀释 → 画质下降(4步蒸馏)。多段 seamless 仍是默认质量优先路径。
        if single_pass and 生成模式 == SEGMENT_MODE_ONE_SHOT and backend == BACKEND_SCAIL2:
            full_start = min(s for s, _, _ in scenes) if scenes else 0
            full_end = max(e for _, e, _ in scenes) if scenes else 0
            full_duration = max(0, full_end - full_start)
            if full_duration <= 0:
                node_error("Planner", "方案C 单遍失败：视频时长无效")
                return []
            target_frames = max(9, ((full_duration - 1) // 4) * 4 + 1)
            complexity = self._estimate_complexity(full_duration, target_fps)
            if 自适应参数:
                _, steps, cfg = self._adaptive_params(complexity, 每段最大帧数)
            else:
                steps, cfg = 30, 6.0
            prompt = prompts[0] if prompts else ""
            segments.append(SegmentInfo(
                index=0,
                start_frame=full_start,
                end_frame=full_end,
                target_frames=target_frames,
                overlap_prev=0,
                overlap_next=0,
                complexity=complexity,
                ref_strategy=REF_STRATEGY_USER_IMAGE,
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
            return segments

        for scene_idx, scene in enumerate(scenes):
            start, end, duration = scene
            complexity = self._estimate_complexity(duration, target_fps)

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
        return segments


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
            "one_shot": "一镜到底",
            "smart_split": "智能分段",
            "sliding_window": "滑动窗口",
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


def replan_for_backend(plan, new_backend):
    """将已有 plan 按新后端重新切分（保留场景边界/提示词/分辨率/帧率/负面提示）。

    用于「用户切了执行后端、但 plan 是按另一后端规划的」场景：自动按新后端的
    官方分块规则（SCAIL-2=81帧/重叠5；骨骼=4k+1/重叠默认）重切，避免段边界
    跳帧/重影，同时让单一开关即可切换路线，不必手动同步两个「生成后端」widget。
    """
    if new_backend == BACKEND_SCAIL2:
        每段最大帧数, 重叠帧数 = 81, 5
    else:
        每段最大帧数, 重叠帧数 = 81, 8

    res = plan.resolution or [832, 480]
    width, height = (int(x) for x in res[:2])
    prompts = [s.prompt for s in plan.segments]
    负面 = ""
    if plan.segments and getattr(plan.segments[0], "negative_prompt", ""):
        负面 = plan.segments[0].negative_prompt

    # 把每个已有 segment 当作一个场景，按新后端规则重新开窗
    scenes = [(s.start_frame, s.end_frame, s.end_frame - s.start_frame) for s in plan.segments]
    tmp = YunjiiSegmentPlanner()
    segments = tmp._build_segments(
        scenes, new_backend, 每段最大帧数, 重叠帧数, width, height,
        prompts, plan.mode, plan.target_fps, 自适应参数=True, 负面提示词=负面,
    )
    return SegmentPlan(
        mode=plan.mode,
        total_segments=len(segments),
        resolution=[width, height],
        target_fps=plan.target_fps,
        segments=segments,
        backend=new_backend,
    )
