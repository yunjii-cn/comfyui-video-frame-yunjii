import os
import uuid
import json
import time
import logging
import traceback
import urllib.request
import urllib.error
import cv2
import numpy as np
import folder_paths

from .types import (
    SegmentPlan, SegmentResult, SegmentContext,
    SEGMENT_MODE_ONE_SHOT, REF_STRATEGY_PREV_LAST_FRAME,
    BACKEND_WANVIDEO, BACKEND_SCAIL2,
    CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START,
    CONTINUITY_LABEL_TO_VALUE,
    SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO,
    SEAMLESS_PLAN_LABEL_TO_VALUE,
)
from .adapters.direct import DirectAdapter
from .adapters.scail import SCAILAdapter
from .adapters.animateplus import AnimatePlusSCAILAdapter
from .checkpoint import CheckpointManager
from .pipeline import build_pipeline, EffectPipeline
from .effects.base import EffectContext
from .debug_log import node_start, node_end, node_error, debug, info, warn, error

logger = logging.getLogger(__name__)

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SCAIL-2 路线默认工作流（官方 WanVideoWrapper 的 SCAIL embeds 子流程，方案 B）
SCAIL_WORKFLOW_DEFAULT = os.path.join(
    PLUGIN_ROOT, "workflows", "SCAIL2_embed子流程_官方_20260728_0230.json"
)
SCAIL_NODE_MARKER = "WanVideoAddSCAILReferenceEmbeds"

# WanAnimatePlus SCAIL_2 家族标记（Tier 2 暖启动路线）
AP_NODE_MARKER = "WanAnimatePlus SCAIL_2 Embeds"
# Tier 2 内置参考工作流（暖启动可不提供模板，自动用此文件）。
# 来源：用户云集智能目录下同名丝滑工作流（含 WanAnimatePlus SCAIL_2 Embeds 节点），
# 复制为 ASCII 命名以便稳定引用；prepare_workflow 会按需转 API 格式。
AP_WORKFLOW_DEFAULT = os.path.join(
    PLUGIN_ROOT, "workflows", "Tier2_WanAnimatePlus_SCAIL2_template.json"
)


class YunjiiSegmentRunner:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("执行结果", "执行日志", "完成状态")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "段落计划": ("STRING", {"default": "", "tooltip": "来自分段规划器的段落计划JSON"}),
                "工作流模板": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "ComfyUI工作流JSON模板，可直接粘贴JSON或输入.json文件路径"}),
                "执行模式": (
                    ["执行", "仅规划", "续跑"],
                    {"default": "执行", "tooltip": "执行=完整运行; 仅规划=只输出计划; 续跑=从断点继续"},
                ),
                "生成后端": (
                    ["骨骼路线(WanVideo)", "SCAIL-2 路线"],
                    {"default": "骨骼路线(WanVideo)", "tooltip": "骨骼路线=现有WanVideo分段链式; SCAIL-2路线=无骨架端到端动作迁移(需SCAIL-2节点与14B权重)"},
                ),
                "最大重试": ("INT", {"default": 3, "min": 0, "max": 10}),
            },
            "optional": {
                "视频路径": ("STRING", {"default": "", "tooltip": "参考视频路径（从运动分析节点连线传入）"}),
                "参考图": ("IMAGE", {"tooltip": "参考图（从LoadImage节点连线传入），优先使用"}),
                "姿态图": ("IMAGE", {"tooltip": "姿态引导图（从VideoPoseExtractor节点连线传入）"}),
                "人物参考图": ("STRING", {"default": "", "tooltip": "参考图文件名（input目录下），连线传入参考图时此项可忽略"}),
                "起始段": ("INT", {"default": 0, "min": 0, "max": 100}),
                "效果模块": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "可选效果管线模块列表(JSON数组或逗号分隔)，如 [\"mimic\"]。为空=不启用任何效果，行为与现状完全一致。支持: mimic, cinematic, enhance, creative（cinematic 可设 xfade 高级转场）"}),
                "连贯策略": (
                    ["多段无缝(默认)", "单遍连贯(方案C)", "暖启动(Tier2)"],
                    {"default": "多段无缝(默认)",
                     "tooltip": "生成侧时序连续性方案(与拼接模式正交)：多段无缝=标准SCAIL真骨架每段独立高保真生成(推荐,配合Stitcher『真·一镜到底(潜空间拼接)』可得连贯自然接缝); 单遍连贯(方案C)=整片一次去噪但长视频画质软; 暖启动(Tier2)=WanAnimatePlus多段+上段真实帧喂回prefix_frames——实测模仿力与画质弱于多段无缝,非推荐。需对应工作流模板支持"},
                ),
                "模型精度": (
                    ["fp8", "fp16"],
                    {"default": "fp8",
                     "tooltip": "SCAIL-2 路线模型精度：fp8(默认,省显存,RTX3090稳定); fp16(更精细,吃显存,需对应权重文件)。仅 SCAIL-2 路线生效"},
                ),
                "生成质量模式": (
                    ["标准 SCAIL 真骨架（推荐）", "WanAnimatePlus 暖启动(Tier2)"],
                    {"default": "标准 SCAIL 真骨架（推荐）",
                     "tooltip": "显式强制生成质量路线(与『连贯策略』正交)：标准 SCAIL 真骨架=每段独立按真骨架(逐帧姿态)高保真生成,配合Stitcher『真·一镜到底(潜空间拼接)』得连贯自然接缝(推荐,默认); WanAnimatePlus 暖启动(Tier2)=上段末帧喂回prefix续写——实测模仿力与画质弱于标准真骨架,非推荐。标准真骨架模式下 WanAnimatePlus 模板也走真骨架多段(禁用暖启动续写),不会退化成 Tier2"},
                ),
            },
        }

    def run(self, 段落计划, 工作流模板, 执行模式, 最大重试, 生成后端="骨骼路线(WanVideo)", 视频路径="", 参考图=None, 姿态图=None, 人物参考图="", 起始段=0, 效果模块="", ComfyUI地址="127.0.0.1:8188", 连贯策略="", 模型精度="fp8", 生成质量模式="标准 SCAIL 真骨架（推荐）", 无缝连贯方案=SEAMLESS_PLAN_AUTO):
        node_start("Runner", 执行模式=执行模式, 最大重试=最大重试, ComfyUI地址=ComfyUI地址)
        info("Runner", "参数诊断: 段落计划类型=%s, 工作流模板类型=%s, 视频路径='%s', 参考图类型=%s, 姿态图类型=%s, 人物参考图='%s', 起始段=%s, ComfyUI地址='%s'",
             type(段落计划).__name__, type(工作流模板).__name__,
             视频路径,
             type(参考图).__name__ if 参考图 is not None else "None",
             type(姿态图).__name__ if 姿态图 is not None else "None",
             人物参考图, 起始段, ComfyUI地址)

        if not 段落计划.strip():
            node_error("Runner", "未提供段落计划")
            return ("", "⚠ 未提供段落计划", False)

        try:
            plan = SegmentPlan.from_json(段落计划)
        except Exception as e:
            return ("", f"⚠ 解析段落计划失败: {e}", False)

        # 后端一致性校验：plan 中记录的 backend 必须与本次执行选择的后端一致，
        # 否则分段规则（骨骼路线 4k+1 vs SCAIL-2 81/5/76）不匹配会导致画面崩溃。
        # 不一致时不再硬性报错，而是**按执行后端自动重规划**（提取 Planner 开窗逻辑），
        # 这样用户只需在 Imitator 切一个「生成后端」开关即可切换路线，不必手动同步
        # 上游 Planner 的「生成后端」widget（旧设计的两个独立 widget 容易造成不一致）。
        plan_backend = getattr(plan, "backend", BACKEND_WANVIDEO)
        exec_backend = BACKEND_SCAIL2 if 生成后端 == "SCAIL-2 路线" else BACKEND_WANVIDEO
        if plan_backend != exec_backend:
            plan_label = "SCAIL-2 路线" if plan_backend == BACKEND_SCAIL2 else "骨骼路线(WanVideo)"
            exec_label = "SCAIL-2 路线" if exec_backend == BACKEND_SCAIL2 else "骨骼路线(WanVideo)"
            warn("Runner", "后端不一致：计划按[%s]规划，本次执行[%s]，将自动按执行后端重规划",
                 plan_label, exec_label)
            try:
                from .planner import replan_for_backend
                plan = replan_for_backend(plan, exec_backend)
                info("Runner", "已自动重规划为[%s]，共 %d 段", exec_label, plan.total_segments)
            except Exception as e:
                node_error("Runner", f"后端不一致且自动重规划失败: {e}")
                return ("", f"⚠ 后端不匹配且自动重规划失败：{e}\n"
                            f"请确认「分段规划器」与「完美模仿」节点的生成后端选择一致。", False)

        # —— 连贯策略归一（与拼接模式正交的生成侧时序连续性方案）——
        # 优先用显式传入的 连贯策略 参数；为空则从计划内 continuity_strategy 推导；
        # 仍为空则默认多段无缝。中文标签 → 英文值。
        _strategy = 连贯策略 or getattr(plan, "continuity_strategy", "") or CONTINUITY_MULTI_SEG
        _strategy = CONTINUITY_LABEL_TO_VALUE.get(_strategy, _strategy)
        if _strategy not in (CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START):
            _strategy = CONTINUITY_MULTI_SEG
        _precision = 模型精度 or getattr(plan, "model_precision", "") or "fp8"
        if _precision not in ("fp8", "fp16"):
            _precision = "fp8"

        # —— 无缝连贯方案（A/B/C）归一：用户选无缝档位的主入口，优先于旧连贯策略 ——
        # 三档共用真·无缝机制(context滑窗+跨段reference_latent续写)，仅目标时长/防漂移增强不同。
        # A/B 均等价于原『标准 SCAIL 真骨架(推荐)』续写机制；C=单遍(方案C兜底)。
        # 写回 plan 字段，使下游适配器/拼接按方案生效（即使 planner 未跑、直接拿旧 plan 也能生效）。
        _seamless = 无缝连贯方案 or getattr(plan, "seamless_plan", "") or SEAMLESS_PLAN_AUTO
        _seamless = SEAMLESS_PLAN_LABEL_TO_VALUE.get(_seamless, _seamless)
        if _seamless not in (SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO):
            _seamless = SEAMLESS_PLAN_AUTO
        if _seamless == SEAMLESS_PLAN_A:
            _strategy = CONTINUITY_MULTI_SEG
            plan.seamless_plan = SEAMLESS_PLAN_A
            plan.long_video_mode = False
            info("Runner", "无缝连贯方案=A (标准多段无缝, 一般时长≤15s)")
        elif _seamless == SEAMLESS_PLAN_B:
            _strategy = CONTINUITY_MULTI_SEG
            plan.seamless_plan = SEAMLESS_PLAN_B
            plan.long_video_mode = True
            info("Runner", "无缝连贯方案=B (超长视频无缝, 长程防漂移启用)")
        elif _seamless == SEAMLESS_PLAN_C:
            _strategy = CONTINUITY_SINGLE_PASS
            plan.seamless_plan = SEAMLESS_PLAN_C
            plan.long_video_mode = False
            info("Runner", "无缝连贯方案=C (单遍连贯·旧方案C兜底)")
        else:
            # auto：沿用连贯策略已归一出的 _strategy，并同步 plan.seamless_plan 供下游识别
            plan.seamless_plan = _seamless
        if _strategy == CONTINUITY_MULTI_SEG:
            plan.long_video_mode = getattr(plan, "long_video_mode", False)
        info("Runner", "连贯策略=%s, 无缝连贯方案=%s, 长视频模式=%s, 模型精度=%s",
             _strategy, plan.seamless_plan, getattr(plan, "long_video_mode", False), _precision)

        # 效果管线：生成前对每段 prompt / params 应用已启用模块。
        # 空「效果模块」→ 空管线 → 透传，输出与现状完全一致（零回归）。
        effect_pipeline = build_pipeline(效果模块)
        if not effect_pipeline.is_empty:
            ctx = EffectContext(
                prompts=[s.prompt for s in plan.segments],
                params=[s.params for s in plan.segments],
                metadata={"backend": plan.backend},
            )
            new_prompts = effect_pipeline.transform_prompts([s.prompt for s in plan.segments], ctx)
            new_params = effect_pipeline.transform_params([s.params for s in plan.segments], ctx)
            for i, seg in enumerate(plan.segments):
                if i < len(new_prompts):
                    seg.prompt = new_prompts[i]
                if i < len(new_params):
                    seg.params = new_params[i]
            info("Runner", "已应用效果模块: %s", effect_pipeline.describe())

        if 执行模式 == "仅规划":
            summary = f"📋 仅规划模式，共 {plan.total_segments} 段\n"
            for seg in plan.segments:
                summary += f"  段{seg.index}: 帧{seg.start_frame}-{seg.end_frame}, {seg.target_frames}帧, 参考={seg.ref_strategy}\n"
            return (段落计划, summary, True)

        template_text = (工作流模板 or "").strip()
        # 若是文件路径，先读成内容；后续所有家族判定都基于内容而非路径
        # （否则 WanAnimatePlus 模板在『路径字符串』阶段会被误判，导致暖启动提前报错）。
        if template_text and os.path.isfile(template_text):
            try:
                with open(template_text, "r", encoding="utf-8") as f:
                    template_text = f.read()
                info("Runner", "从文件加载工作流模板: %s", 工作流模板.strip())
            except Exception as e:
                return ("", f"⚠ 无法读取模板文件 {工作流模板}: {e}", False)
        # 模板所含 SCAIL 家族判定（基于内容）
        is_ap_template = (AP_NODE_MARKER in template_text)
        is_std_scail_template = (SCAIL_NODE_MARKER in template_text)

        if 生成后端 == "SCAIL-2 路线":
            # 未提供模板或提供的不是 SCAIL 工作流时，按策略选内置默认：
            # 暖启动(Tier2) 优先用 WanAnimatePlus 参考工作流；否则用标准 SCAIL 子流程。
            if not (is_ap_template or is_std_scail_template):
                if _strategy == CONTINUITY_WARM_START and os.path.isfile(AP_WORKFLOW_DEFAULT):
                    try:
                        with open(AP_WORKFLOW_DEFAULT, "r", encoding="utf-8") as f:
                            template_text = f.read()
                        info("Runner", "暖启动(Tier2): 使用内置 WanAnimatePlus 参考工作流 %s", AP_WORKFLOW_DEFAULT)
                    except Exception as e:
                        return ("", f"⚠ 无法读取内置 Tier2 模板 {AP_WORKFLOW_DEFAULT}: {e}", False)
                    is_ap_template = True
                elif os.path.isfile(SCAIL_WORKFLOW_DEFAULT):
                    try:
                        with open(SCAIL_WORKFLOW_DEFAULT, "r", encoding="utf-8") as f:
                            template_text = f.read()
                        info("Runner", "SCAIL-2 路线：使用内置官方工作流 %s", SCAIL_WORKFLOW_DEFAULT)
                    except Exception as e:
                        return ("", f"⚠ 无法读取内置 SCAIL 模板 {SCAIL_WORKFLOW_DEFAULT}: {e}", False)
                    is_std_scail_template = True
                elif not template_text:
                    return ("", "⚠ SCAIL-2 路线缺少工作流模板，请粘贴 ComfyUI 工作流JSON或输入JSON文件路径", False)
            # 重判模板家族（若上面填了默认，或用户直接粘贴内容）
            is_ap_template = (AP_NODE_MARKER in template_text)
            is_std_scail_template = (SCAIL_NODE_MARKER in template_text)
            # 显式『生成质量模式』开关：默认『标准 SCAIL 真骨架』= 强制真骨架多段
            # (禁用暖启动续写)。仅当模板为 WanAnimatePlus 家族且用户显式选
            # 『WanAnimatePlus 暖启动(Tier2)』时才允许暖启动；用户真实 WanAnimatePlus
            # 模板选默认即走真骨架多段(高保真,非 Tier2 续写)。
            _quality = 生成质量模式 or "标准 SCAIL 真骨架（推荐）"
            if _quality == "标准 SCAIL 真骨架（推荐）" and is_ap_template:
                _strategy = CONTINUITY_MULTI_SEG
            if _strategy == CONTINUITY_WARM_START and not (is_ap_template or is_std_scail_template):
                # 暖启动两大家族均支持：
                #  · WanAnimatePlus SCAIL_2 → prefix_frames 帧级硬冻结暖启动
                #  · 标准 WanVideoWrapper SCAIL → WanVideoSamplerv2.samples latent 暖启动(D-A 方案)
                # 两者都不是(非 SCAIL 模板)才拒绝，给出明确指引而非静默降级。
                return ("", "⚠ 暖启动需要一个『SCAIL-2』工作流模板"
                                "（WanAnimatePlus 含 prefix_frames 入口，或标准 WanVideoWrapper SCAIL 走 latent 暖启动）。"
                                "请在『工作流模板』中粘贴对应工作流 JSON 或其文件路径，"
                                "或确认已放置内置参考工作流。", False)
            if _strategy == CONTINUITY_SINGLE_PASS and is_ap_template:
                warn("Runner", "单遍连贯(方案C) 在 WanAnimatePlus 家族不支持，回退为『多段无缝』运行该模板")
        elif not template_text:
            return ("", "⚠ 未提供工作流模板，请粘贴ComfyUI工作流JSON或输入JSON文件路径", False)
        if os.path.isfile(template_text):
            try:
                with open(template_text, "r", encoding="utf-8") as f:
                    template_text = f.read()
                info("Runner", "从文件加载工作流模板: %s", 工作流模板.strip())
            except Exception as e:
                return ("", f"⚠ 无法读取模板文件 {template_text}: {e}", False)

        try:
            workflow_raw = json.loads(template_text)
        except json.JSONDecodeError as e:
            return ("", f"⚠ 工作流模板JSON格式错误: {e}", False)

        if 生成后端 == "SCAIL-2 路线":
            # SCAIL-2 路线含两大家族，按模板自动选适配器：
            #  · WanAnimatePlus SCAIL_2  → AnimatePlusSCAILAdapter（支持潜空间拼接；
            #    『标准 SCAIL 真骨架』模式下强制 multi_seg 真骨架多段，禁用暖启动续写）
            #  · 标准 WanVideoWrapper SCAIL → SCAILAdapter（多段 / 单遍方案C）
            if is_ap_template:
                gen_adapter = AnimatePlusSCAILAdapter(folder_paths.get_output_directory())
                if _quality == "标准 SCAIL 真骨架（推荐）":
                    backend_name = "SCAIL-2 路线(WanAnimatePlus/真骨架多段)"
                else:
                    backend_name = "SCAIL-2 路线(WanAnimatePlus/Tier2暖启动)"
            else:
                gen_adapter = SCAILAdapter(folder_paths.get_output_directory())
                backend_name = "SCAIL-2 路线"
            gen_adapter.driving_video_path = 视频路径  # SCAIL 用源视频作动作驱动
            gen_adapter.model_precision = _precision
            workflow = gen_adapter.prepare_workflow(workflow_raw)
            if not workflow:
                return ("", "⚠ SCAIL-2 工作流预处理失败（节点缺失或格式无法识别）", False)
            info("Runner", "%s 工作流预处理完成: %d个节点", backend_name, len(workflow))
        else:
            workflow = self._ensure_api_format(workflow_raw)
            if not workflow:
                return ("", "⚠ 工作流模板格式无法识别，请提供API格式或完整工作流格式", False)
            gen_adapter = DirectAdapter(folder_paths.get_output_directory())
            backend_name = "骨骼路线(WanVideo)"
        info("Runner", "使用生成后端: %s（内联PromptExecutor）", backend_name)
        info("Runner", "工作流格式: API格式, %d个节点", len(workflow))
        node_map = gen_adapter.discover_nodes(workflow)
        info("Runner", "节点发现结果: %s", node_map.to_dict())
        info("Runner", "工作流节点列表: %s", list(workflow.keys()))

        if not node_map.is_valid():
            return ("", f"⚠ 工作流中缺少后端【{backend_name}】所需的必要节点。已发现: {node_map.to_dict()}", False)

        log_lines = []
        run_id = time.strftime("%Y%m%d_%H%M%S")
        log_lines.append(f"🚀 开始链式执行: {plan.total_segments}段, 模式={plan.mode}, run_id={run_id}")
        log_lines.append(f"📍 发现节点: {node_map.to_dict()}")
        log_lines.append(f"📥 输入诊断: 视频路径='{视频路径}', 参考图={'已连接(IMAGE)' if 参考图 is not None else '未连接'}, 姿态图={'已连接(IMAGE)' if 姿态图 is not None else '未连接'}, 人物参考图='{人物参考图}'")
        info("Runner", "run_id=%s, 视频路径='%s', 参考图=%s, 姿态图=%s, 人物参考图='%s'", run_id,
             视频路径,
             "IMAGE已连接" if 参考图 is not None else "None",
             "IMAGE已连接" if 姿态图 is not None else "None",
             人物参考图)
        log_lines.append("")

        if 人物参考图 and 人物参考图.strip() and not any(c in 人物参考图 for c in '.-_'):
            try:
                int(人物参考图.strip())
                info("Runner", "检测到widgets_values映射错位: 人物参考图='%s' (应为文件名), 自动修正为空", 人物参考图)
                人物参考图 = ""
            except ValueError:
                pass

        ref_image_path = ""
        if 参考图 is not None:
            ref_image_path = self._save_ref_image(参考图)
            if ref_image_path:
                log_lines.append(f"👤 参考图（从连线获取）: {os.path.basename(ref_image_path)}")
                info("Runner", "参考图（从连线获取）: %s", os.path.basename(ref_image_path))
        else:
            warn("Runner", "参考图未通过IMAGE链接传入(=None)，请检查工作流中Node30(LoadImage)→Node21(Runner)的连线是否正确")

        if not ref_image_path and 人物参考图.strip():
            input_dir = folder_paths.get_input_directory()
            candidate = os.path.join(input_dir, 人物参考图.strip())
            if os.path.isfile(candidate):
                ref_image_path = candidate
                log_lines.append(f"👤 用户参考图: {人物参考图}")
                info("Runner", "用户参考图: %s", 人物参考图)

        if not ref_image_path:
            log_lines.append("⚠️ 未提供参考图，将使用模板默认图片")
            warn("Runner", "未提供参考图，将使用模板默认图片")

        pose_dir = ""
        if 姿态图 is not None:
            pose_dir = self._save_pose_images(姿态图)
            if pose_dir:
                log_lines.append(f"🕺 姿态图（从连线获取）: {os.path.basename(pose_dir)}")
                info("Runner", "姿态图（从连线获取）: %s", pose_dir)
        else:
            warn("Runner", "姿态图未通过IMAGE链接传入(=None)，请检查工作流中Node2(PoseExtractor)→Node21(Runner)的连线是否正确")

        cp = CheckpointManager(plan.mode)
        if 执行模式 == "续跑":
            cp_data = cp.load()
            if cp_data:
                起始段 = cp_data.get("current_segment", 起始段)
                prev_frame = cp_data.get("prev_last_frame", "")
                if prev_frame and os.path.isfile(prev_frame):
                    ref_image_path = prev_frame
                log_lines.append(f"🔄 续跑模式: 从段{起始段}继续")

        results = []
        prev_context = SegmentContext(last_frame_path=ref_image_path)
        all_success = True
        prev_video_path = ""  # 上一段成片视频路径（暖启动 Tier2 用于 pixel prefix 注入）
        prev_latent_path = ""  # 上一段落盘 latent 路径（根治 方案C：latent 视频续写跨段共享上下文）

        start_from = 起始段 if 起始段 > 0 else 0

        info("Runner", "开始链式执行: %d段, 节点映射=%s", plan.total_segments, node_map.to_dict())

        gen_adapter.init_executor()
        try:
            for seg in plan.segments:
                if seg.index < start_from:
                    continue

                seg_start_time = time.time()
                log_lines.append(f"▶ 段{seg.index}/{plan.total_segments - 1}: 帧{seg.start_frame}-{seg.end_frame}, {seg.target_frames}帧")
                info("Runner", "▶ 开始段%d/%d: 帧%d-%d, %d帧", seg.index, plan.total_segments - 1,
                     seg.start_frame, seg.end_frame, seg.target_frames)

                current_ref = ""
                if seg.ref_strategy == REF_STRATEGY_PREV_LAST_FRAME and prev_context.last_frame_path:
                    current_ref = prev_context.last_frame_path
                    log_lines.append(f"  🔗 使用前段末帧作为参考图")
                elif ref_image_path and os.path.isfile(ref_image_path):
                    current_ref = ref_image_path
                    log_lines.append(f"  👤 使用用户参考图")

                # 根治(方案C)：跨段 latent 上下文共享，从架构层消除动作相位断裂。
                # 触发条件：
                #   · 暖启动(Tier2)：沿用上段成片做 pixel prefix 注入（原行为）；
                #   · 标准 SCAIL 真骨架(推荐)：对 seg>0 启用 latent 视频续写——
                #     SCAILAdapter 用上段视频重编码 latent；AnimatePlus 直接加载上段
                #     落盘 latent，二者均在采样时与上段动作共享 latent 上下文。
                # 无缝连贯方案（A/B/C）→ 启用生成侧连续续写(context_options 跨段 reference_latent)。
                # 等价于原『标准 SCAIL 真骨架(推荐)』机制：对 seg>0 共享上段 latent 上下文。
                # C 方案走单遍(_strategy=single_pass，整片一次去噪，不进此多段循环)。
                _seamless_on = (
                    (plan.seamless_plan in (SEAMLESS_PLAN_A, SEAMLESS_PLAN_B))
                    or (_quality == "标准 SCAIL 真骨架（推荐）")
                    or (_strategy == CONTINUITY_MULTI_SEG)
                )
                _prev_vp = prev_video_path if (seg.index > 0 and _strategy == CONTINUITY_WARM_START) else ""
                _latent_warmstart = _seamless_on and seg.index > 0
                wf = gen_adapter.modify_workflow_for_segment(
                    workflow, node_map, seg, current_ref, pose_dir, run_id,
                    user_ref_path=ref_image_path, prev_video_path=_prev_vp,
                    prev_latent_path=(prev_latent_path if _latent_warmstart else ""),
                    latent_warmstart=_latent_warmstart)

                success = False
                last_error = ""

                for attempt in range(最大重试 + 1):
                    try:
                        info("Runner", "段%d: 内联执行 (尝试%d/%d)", seg.index, attempt + 1, 最大重试 + 1)

                        # 主输出节点：骨骼路线=NodeMap.video_combine，SCAIL路线=SCAILNodeMap.combine。
                        # 传给适配器，使其在多个视频输出节点时优先抓取真实成片(而非姿态/预览骨架视频)。
                        primary_out = getattr(node_map, "video_combine", None) or getattr(node_map, "combine", None)
                        result = gen_adapter.execute_inline(wf, timeout=3600, primary_output_node=primary_out)
                        status = result.get("status", "")

                        if status == "success":
                            output_path = result.get("video_path", "")
                            last_frame = ""
                            if output_path and os.path.isfile(output_path):
                                last_frame = self._extract_last_frame(output_path, run_id)

                            seg_duration = time.time() - seg_start_time
                            # 潜空间拼接：若适配器支持 latent 落盘(标准 SCAIL / AnimatePlus 均继承)，
                            # 取与适配器一致的确定性路径；stitcher 在 latent_blend 模式下消费，缺失则回退像素拼接。
                            latent_path = ""
                            if run_id and hasattr(gen_adapter, "_latent_save_path"):
                                try:
                                    latent_path = gen_adapter._latent_save_path(run_id, seg.index)
                                except Exception:
                                    latent_path = ""
                            seg_result = SegmentResult(
                                segment_index=seg.index,
                                video_path=output_path,
                                last_frame_path=last_frame,
                                status="success",
                                prompt_id=result.get("prompt_id", ""),
                                duration_sec=seg_duration,
                                overlap_prev=seg.overlap_prev,
                                latent_path=latent_path,
                            )
                            results.append(seg_result)

                            if last_frame:
                                prev_context = SegmentContext(last_frame_path=last_frame)
                                cp.save(seg.index, last_frame, [r.to_dict() for r in results])

                            # 记录上段成片 / 落盘 latent，供跨段连续性使用：
                            #  · prev_video_path → Tier2 像素 prefix 注入
                            #  · prev_latent_path → 根治 方案C latent 视频续写(共享上段 latent 上下文)
                            if output_path and os.path.isfile(output_path):
                                prev_video_path = output_path
                            if latent_path and os.path.isfile(latent_path):
                                prev_latent_path = latent_path

                            log_lines.append(f"  ✅ 完成! 耗时{seg_duration:.1f}s, 输出={output_path}")
                            info("Runner", "段%d: ✅ 完成! 耗时%.1fs", seg.index, seg_duration)
                            success = True
                            break
                        elif status == "timeout":
                            last_error = f"超时(>{1800}s)"
                            log_lines.append(f"  ⏰ 超时(尝试{attempt + 1}/{最大重试 + 1})")
                            warn("Runner", "段%d: 超时!", seg.index)
                        else:
                            last_error = result.get("error", "unknown error")
                            log_lines.append(f"  ❌ 失败(尝试{attempt + 1}/{最大重试 + 1}): {last_error[:200]}")
                            warn("Runner", "段%d: 失败 - %s", seg.index, last_error[:300])

                    except Exception as e:
                        last_error = str(e)
                        log_lines.append(f"  ❌ 异常(尝试{attempt + 1}/{最大重试 + 1}): {last_error[:100]}")
                        warn("Runner", "段%d: 异常 - %s", seg.index, last_error[:100])

                if not success:
                    all_success = False
                    results.append(SegmentResult(
                        segment_index=seg.index,
                        status="failed",
                        error=last_error,
                    ))
                    log_lines.append(f"  🛑 段{seg.index}最终失败: {last_error[:200]}")
                    error("Runner", "段%d: 最终失败 - %s", seg.index, last_error[:200])

                    if prev_context.last_frame_path and os.path.isfile(prev_context.last_frame_path):
                        log_lines.append(f"  ⏭ 跳过此段，继续使用上一段末帧")

                log_lines.append("")
        finally:
            gen_adapter.cleanup_executor()

        results_json = json.dumps(
            {"run_id": run_id, "mode": plan.mode, "segments": [r.to_dict() for r in results]},
            ensure_ascii=False, indent=2
        )

        total = len(results)
        ok = sum(1 for r in results if r.status == "success")
        log_lines.append(f"{'✅' if all_success else '⚠️'} 执行完毕: {ok}/{total}段成功")

        info("Runner", "执行完毕: %d/%d段成功", ok, total)
        node_end("Runner", f"{ok}/{total}段成功")

        return (results_json, "\n".join(log_lines), all_success)

    @staticmethod
    def _ensure_api_format(workflow_raw):
        if not isinstance(workflow_raw, dict):
            return None

        first_key = next(iter(workflow_raw), None)
        if first_key is not None and isinstance(workflow_raw[first_key], dict):
            if "class_type" in workflow_raw[first_key] or "type" in workflow_raw[first_key]:
                return workflow_raw

        if "nodes" in workflow_raw and "links" in workflow_raw:
            converted = YunjiiSegmentRunner._convert_full_to_api(workflow_raw)
            info("Runner", "工作流从完整格式转换为API格式: %d个节点", len(converted))
            return converted

        return None

    @staticmethod
    def _convert_full_to_api(full_workflow):
        nodes_list = full_workflow.get("nodes", [])
        links_list = full_workflow.get("links", [])

        link_map = {}
        for link in links_list:
            if len(link) >= 5:
                link_id = link[0]
                src_node = str(link[1])
                src_slot = link[2]
                dst_node = str(link[3])
                dst_slot = link[4]
                link_map[link_id] = {
                    "src_node": src_node,
                    "src_slot": src_slot,
                    "dst_node": dst_node,
                    "dst_slot": dst_slot,
                }

        node_class_mappings = YunjiiSegmentRunner._get_node_class_mappings()

        # mode==4 的节点在 ComfyUI 里是『禁用/绕过(bypass)』，不执行且输出视为断开。
        # 若消费方仍引用它的输出连线，转换后会指向不存在的节点 -> 执行报 NodeNotFoundError。
        # 这里跳过这类连线（消费方输入视作未连接），与 _drop_bypassed 语义一致。
        bypassed_ids = {str(n.get("id")) for n in nodes_list if (n.get("mode", 0) == 4)}

        api_workflow = {}
        for node in nodes_list:
            node_id = str(node.get("id", ""))
            class_type = node.get("type", "")
            mode = node.get("mode", 0)
            if mode == 4:
                continue

            inputs_def = node.get("inputs", [])
            widgets_values = node.get("widgets_values", [])

            linked_inputs = set()
            api_inputs = {}

            for inp in inputs_def:
                inp_name = inp.get("name", "")
                inp_type = inp.get("type", "")
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    link_info = link_map[link_id]
                    if link_info["src_node"] in bypassed_ids:
                        continue
                    api_inputs[inp_name] = [link_info["src_node"], link_info["src_slot"]]
                    linked_inputs.add(inp_name)

            widget_names = YunjiiSegmentRunner._get_widget_names(class_type, node_class_mappings)

            widget_idx = 0
            for wval in widgets_values:
                if widget_idx < len(widget_names):
                    wname = widget_names[widget_idx]
                    if wname not in linked_inputs:
                        api_inputs[wname] = wval
                widget_idx += 1

            api_workflow[node_id] = {
                "class_type": class_type,
                "inputs": api_inputs,
            }

        return api_workflow

    @staticmethod
    def _get_node_class_mappings():
        try:
            import nodes
            return nodes.NODE_CLASS_MAPPINGS
        except Exception:
            pass
        return {}

    @staticmethod
    def _get_widget_names(class_type, node_class_mappings):
        node_cls = node_class_mappings.get(class_type)
        if node_cls is None:
            return []

        try:
            input_types = node_cls.INPUT_TYPES()
            widget_names = []

            for category in ["required", "optional"]:
                cat_inputs = input_types.get(category, {})
                for name, config in cat_inputs.items():
                    if isinstance(config, (list, tuple)) and len(config) > 0:
                        typ = config[0]
                        # widget 输入：标量类型(STRING/INT/...) 或 COMBO(选项列表)
                        is_widget = (
                            isinstance(typ, str)
                            and typ
                            in ["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO", "ENUM"]
                        ) or isinstance(typ, list)
                        if is_widget:
                            widget_names.append(name)

            return widget_names
        except Exception:
            return []

    @staticmethod
    def _save_ref_image(image_tensor, suffix="ref"):
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
            os.makedirs(input_dir, exist_ok=True)

            img = image_tensor[0].cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            fname = f"yunjii_{suffix}_{int(time.time())}.png"
            fpath = os.path.join(input_dir, fname)
            cv2.imwrite(fpath, img_bgr)

            info("Runner", "参考图已保存: %s", fpath)
            return fpath
        except Exception as e:
            logger.error("保存参考图失败: %s", e)
            return ""

    @staticmethod
    def _save_pose_images(pose_tensor):
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
            pose_dir = os.path.join(input_dir, f"yunjii_poses_{int(time.time())}")
            os.makedirs(pose_dir, exist_ok=True)

            num_frames = pose_tensor.shape[0]
            h, w = pose_tensor.shape[1], pose_tensor.shape[2]

            for i in range(num_frames):
                img = pose_tensor[i].cpu().numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                fname = f"pose_{i:05d}.png"
                cv2.imwrite(os.path.join(pose_dir, fname), img_bgr)

            video_path = os.path.join(pose_dir, "poses.mp4")
            fourcc = 0
            for codec in ["mp4v", "avc1", "H264", "XVID"]:
                c = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(os.path.join(pose_dir, "_test.mp4"), c, 16.0, (w, h))
                if test_writer.isOpened():
                    test_writer.release()
                    fourcc = c
                    os.remove(os.path.join(pose_dir, "_test.mp4"))
                    break
            if fourcc == 0:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 16.0, (w, h))
            for i in range(num_frames):
                img = pose_tensor[i].cpu().numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img_bgr)
            writer.release()

            info("Runner", "姿态图已保存: %s (%d帧), 视频: %s", pose_dir, num_frames, video_path)
            return pose_dir
        except Exception as e:
            logger.error("保存姿态图失败: %s", e)
            return ""

    def _extract_last_frame(self, video_path, run_id=""):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return ""
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return ""
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return ""

            output_dir = folder_paths.get_output_directory()
            sub_dir = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
            chain_dir = os.path.join(output_dir, "yunjii_v2v", sub_dir, "chain")
            os.makedirs(chain_dir, exist_ok=True)

            fname = f"seg_lastframe_{int(time.time())}.png"
            path = os.path.join(chain_dir, fname)
            cv2.imwrite(path, frame)
            logger.info("Extracted last frame: %s", path)
            return path
        except Exception as e:
            logger.error("Failed to extract last frame: %s", e)
            return ""
