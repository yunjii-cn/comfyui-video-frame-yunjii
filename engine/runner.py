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
    BACKEND_WANVIDEO, BACKEND_SCAIL2, BACKEND_SCAIL2_NATIVE,
    CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START,
    CONTINUITY_LABEL_TO_VALUE,
    SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO,
    SEAMLESS_PLAN_LABEL_TO_VALUE,
    UNIFIED_PLAN_LABELS, resolve_unified_plan,
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
                    ["骨骼路线(WanVideo)", "SCAIL-2 路线", "原生 SCAIL-2 长视频(一镜到底)"],
                    {"default": "骨骼路线(WanVideo)", "tooltip": "骨骼路线=现有WanVideo分段链式; SCAIL-2路线=无骨架端到端动作迁移(需SCAIL-2节点与14B权重); 原生 SCAIL-2 长视频=直驱 comfyui_scail2_multi_cond 长视频调度节点(一镜到底动作模仿，需已安装该包)"},
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
                "连贯方案": (
                    [label for _, label in UNIFIED_PLAN_LABELS],
                    {"default": "A·标准多段无缝(≤15s) ⭐默认",
                     "tooltip": "一镜到底动作模仿的连贯档位（单一入口，已合并旧『连贯策略』+『无缝连贯方案』两个下拉）：\n"
                                "· A 标准多段无缝(默认)：一般≤15s，多段独立生成、接缝交叉溶解平滑；每段质量最高、可分段重试。\n"
                                "· B 超长视频无缝：15~30s+，单遍连续采样+context滑窗覆盖全帧=真·零接缝、长视频不劣化(⭐长片推荐)；显存峰值更高、不可分段重试。\n"
                                "· C 单遍兜底：整片一次去噪、不注入滑窗，>5s画质软，仅对比/兜底。\n"
                                "· 暖启动(Tier2)：WanAnimatePlus多段+上段真实帧喂回prefix_frames（需SCAIL-2路线+WanAnimatePlus模板）。"},
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

    def run(self, 段落计划, 工作流模板, 执行模式, 最大重试, 生成后端="骨骼路线(WanVideo)", 视频路径="", 参考图=None, 姿态图=None, 人物参考图="", 起始段=0, 效果模块="", ComfyUI地址="127.0.0.1:8188", 模型精度="fp8", 生成质量模式="标准 SCAIL 真骨架（推荐）", 连贯方案=SEAMLESS_PLAN_AUTO):
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
        if 生成后端 == "原生 SCAIL-2 长视频(一镜到底)":
            exec_backend = BACKEND_SCAIL2_NATIVE
        elif 生成后端 == "SCAIL-2 路线":
            exec_backend = BACKEND_SCAIL2
        else:
            exec_backend = BACKEND_WANVIDEO
        if plan_backend != exec_backend:
            if exec_backend == BACKEND_SCAIL2_NATIVE:
                # 原生 SCAIL-2 长视频节点内部自行调度分段，不依赖 yunjii 的段帧规则，
                # 故不触发 replan_for_backend（该接口未实现 native 重规划）。
                plan.backend = BACKEND_SCAIL2_NATIVE
                info("Runner", "原生 SCAIL-2 后端：跳过按段重规划（原生节点内部分段调度）")
            else:
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

        _precision = 模型精度 or getattr(plan, "model_precision", "") or "fp8"
        if _precision not in ("fp8", "fp16"):
            _precision = "fp8"

        # —— 统一「连贯方案」下拉归一（合并旧 连贯策略 / 无缝连贯方案 为一个）——
        # 用户只面对一个选择：A 标准多段无缝 / B 超长视频无缝 / C 单遍兜底 / 暖启动(Tier2)。
        # 写回 plan 字段，使下游适配器/拼接按方案生效（即使 planner 未跑、直接拿旧 plan 也能生效）。
        _strategy_from_unified, _seamless, _mode_u = resolve_unified_plan(连贯方案)
        if _seamless not in (SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO):
            _seamless = SEAMLESS_PLAN_AUTO
        if _strategy_from_unified == CONTINUITY_WARM_START:
            _strategy = CONTINUITY_WARM_START
            plan.seamless_plan = SEAMLESS_PLAN_AUTO
            plan.long_video_mode = False
            info("Runner", "连贯方案=暖启动(Tier2): 分段 + 上段真实帧喂回 WanAnimatePlus prefix_frames")
        elif _seamless == SEAMLESS_PLAN_A:
            _strategy = CONTINUITY_MULTI_SEG
            plan.seamless_plan = SEAMLESS_PLAN_A
            plan.long_video_mode = False
            info("Runner", "连贯方案=A (标准多段无缝, 一般时长≤15s)")
        elif _seamless == SEAMLESS_PLAN_B:
            # B 方案：超长视频无缝 = 单遍连续采样(single_pass 规划) + context 滑窗。
            # 与 C 同为 single_pass+滑窗；差异仅 B 无时长上限、C 超限自动回退多段。
            _strategy = CONTINUITY_SINGLE_PASS
            plan.seamless_plan = SEAMLESS_PLAN_B
            plan.long_video_mode = True
            info("Runner", "连贯方案=B (超长视频无缝: 单遍连续采样+context滑窗, 真·无漂移)")
        elif _seamless == SEAMLESS_PLAN_C:
            _strategy = CONTINUITY_SINGLE_PASS
            plan.seamless_plan = SEAMLESS_PLAN_C
            plan.long_video_mode = False
            info("Runner", "连贯方案=C (单遍+滑窗防劣化, 超上限自动回退多段)")
        else:
            # auto：沿用计划内 continuity_strategy / seamless_plan（兼容旧 plan JSON）
            _strategy = getattr(plan, "continuity_strategy", "") or CONTINUITY_MULTI_SEG
            _strategy = CONTINUITY_LABEL_TO_VALUE.get(_strategy, _strategy)
            if _strategy not in (CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START):
                _strategy = CONTINUITY_MULTI_SEG
            plan.seamless_plan = getattr(plan, "seamless_plan", "") or SEAMLESS_PLAN_AUTO
        if _strategy == CONTINUITY_MULTI_SEG:
            plan.long_video_mode = getattr(plan, "long_video_mode", False)
        info("Runner", "连贯方案=%s, 长视频模式=%s, 模型精度=%s",
             plan.seamless_plan, getattr(plan, "long_video_mode", False), _precision)

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

        # —— 原生 SCAIL-2 长视频后端：直驱 comfyui_scail2_multi_cond 调度节点（一镜到底动作模仿）——
        # 与既有「骨骼/SCAIL-2 路线」(模板工作流 + 分段循环) 完全正交：原生节点内部自行调度分段，
        # 故不走下面的模板预处理/分段循环，单独构造自包含 prompt 后交给内联执行器。
        if 生成后端 == "原生 SCAIL-2 长视频(一镜到底)":
            return self._run_native_scail2(
                plan, 视频路径, 参考图, 人物参考图, _precision, 连贯方案, 执行模式, 最大重试, ComfyUI地址)

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
                # 多段无缝(A)与暖启动(Tier2) 默认用 WanAnimatePlus 家族模板——该家族
                # 原生支持 transition_video 尾帧硬冻结续写（肥猴『分段队列』SQR 同款
                # 接段机制），多段一镜到底的段间连续由此在生成侧硬保证。
                # 单遍连贯(B/C) 仍走标准官方子流程（AP 家族不支持整片单遍去噪）。
                if _strategy in (CONTINUITY_WARM_START, CONTINUITY_MULTI_SEG) and os.path.isfile(AP_WORKFLOW_DEFAULT):
                    try:
                        with open(AP_WORKFLOW_DEFAULT, "r", encoding="utf-8") as f:
                            template_text = f.read()
                        info("Runner", "多段无缝/暖启动: 使用内置 WanAnimatePlus 参考工作流(原生 transition_video 接段) %s", AP_WORKFLOW_DEFAULT)
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

        results = []
        prev_context = SegmentContext(last_frame_path=ref_image_path)
        all_success = True
        prev_video_path = ""  # 上一段成片视频路径（Tier2: transition_video 尾帧硬冻结续写 / latent 暖启动回退）
        prev_latent_path = ""  # 上一段落盘 latent 路径（根治 方案C：latent 视频续写跨段共享上下文）

        cp = CheckpointManager(plan.mode)
        if 执行模式 == "续跑":
            cp_data = cp.load()
            if cp_data:
                起始段 = cp_data.get("current_segment", 起始段)
                prev_frame = cp_data.get("prev_last_frame", "")
                if prev_frame and os.path.isfile(prev_frame):
                    ref_image_path = prev_frame
                # 恢复「上段成片/latent 路径」：续跑的第一段(段号>0)才能继续做
                # transition_video 尾帧硬冻结续写 / latent 续写（否则该段接缝退化为独立生成）。
                try:
                    for _r in (cp_data.get("results") or []):
                        if not isinstance(_r, dict):
                            continue
                        _vp = _r.get("video_path", "")
                        if _vp and os.path.isfile(_vp):
                            prev_video_path = _vp
                        _lp = _r.get("latent_path", "")
                        if _lp and os.path.isfile(_lp):
                            prev_latent_path = _lp
                except Exception:
                    pass
                log_lines.append(f"🔄 续跑模式: 从段{起始段}继续")

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
                # 上段成片路径：seg>0 时对 SCAIL 两大家族适配器都传递——
                # AnimatePlus 家族用它做 transition_video 尾帧硬冻结续写（主路径），
                # 标准家族用它做 latent 暖启动回退；骨骼路线(DirectAdapter)不传，
                # 保持「独立分段 + 拼接淡化」现状。
                _prev_vp = prev_video_path if (seg.index > 0 and isinstance(
                    gen_adapter, (SCAILAdapter, AnimatePlusSCAILAdapter))) else ""
                _latent_warmstart = _seamless_on and seg.index > 0
                wf = gen_adapter.modify_workflow_for_segment(
                    workflow, node_map, seg, current_ref, pose_dir, run_id,
                    user_ref_path=ref_image_path, prev_video_path=_prev_vp,
                    prev_latent_path=(prev_latent_path if _latent_warmstart else ""),
                    latent_warmstart=_latent_warmstart,
                    seamless_plan=plan.seamless_plan)

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

    def _run_native_scail2(self, plan, 视频路径, 参考图, 人物参考图, _precision, 连贯方案, 执行模式, 最大重试, ComfyUI地址):
        """原生 SCAIL-2 长视频后端：构造自包含 prompt 并交给内联执行器（一镜到底动作模仿）。

        与既有「骨骼/SCAIL-2 路线」正交——原生节点内部自行调度分段，不走模板预处理/分段循环。
        未安装 comfyui_scail2_multi_cond 时明确报错，不静默降级。
        """
        node_start("Runner-NativeSCAIL2", ComfyUI地址=ComfyUI地址)
        from .adapters.scail2_native import is_native_scail2_available, build_native_prompt

        if not is_native_scail2_available():
            node_end("Runner-NativeSCAIL2", "未安装原生包")
            return ("", "⚠ 原生 SCAIL-2 后端需要安装 comfyui_scail2_multi_cond 节点包并重启 ComfyUI。"
                          "请先在 custom_nodes 安装该包，再选此后端。", False)

        # 驱动视频（动作来源）
        driving = (视频路径 or "").strip()
        if not driving or not os.path.isfile(driving):
            node_end("Runner-NativeSCAIL2", "缺少驱动视频")
            return ("", "⚠ 原生 SCAIL-2(一镜到底动作模仿)需要驱动视频(动作)路径，"
                          "请在『视频路径』传入源视频。", False)

        # 参考图：优先连线 IMAGE，其次 input 目录文件名
        ref_paths = []
        if 参考图 is not None:
            rp = self._save_ref_image(参考图)
            if rp:
                ref_paths.append(rp)
        if not ref_paths and 人物参考图.strip():
            cand = os.path.join(folder_paths.get_input_directory(), 人物参考图.strip())
            if os.path.isfile(cand):
                ref_paths.append(cand)
        if not ref_paths:
            node_end("Runner-NativeSCAIL2", "缺少参考图")
            return ("", "⚠ 原生 SCAIL-2 需要至少 1 张参考图（连线『参考图』或填『人物参考图』文件名）。", False)

        segs = getattr(plan, "segments", None) or []
        total_frames = sum(int(getattr(s, "target_frames", 0) or 0) for s in segs) or 49

        params = {
            "driving_video": driving,
            "reference_images": ref_paths,
            "seed": 1, "cfg": 1.0, "mode": "replacement",
            "max_frames": int(total_frames),
            "overlap_frames": 5, "reference_count": len(ref_paths),
            "color_correction": True, "cache_mode": "disk",
            "steps": 4, "shift": 5, "sampler_name": "euler_ancestral", "scheduler": "beta",
            "filename_prefix": "yunjii_native_scail2",
        }

        try:
            prompt, out_node = build_native_prompt(plan, params)
        except Exception as e:
            node_end("Runner-NativeSCAIL2", "构造失败")
            return ("", f"⚠ 构造原生 SCAIL-2 工作流失败: {e}", False)

        info("Runner-NativeSCAIL2", "prompt 构造完成: %d 节点, 输出节点=%s, 驱动=%s, 参考=%d张",
             len(prompt), out_node, os.path.basename(driving), len(ref_paths))

        adapter = DirectAdapter(folder_paths.get_output_directory())
        adapter.init_executor()
        try:
            result = adapter.execute_inline(prompt, timeout=3600, primary_output_node=out_node)
        finally:
            adapter.cleanup_executor()

        status = result.get("status", "")
        if status == "success":
            vp = result.get("video_path", "")
            run_id = time.strftime("%Y%m%d_%H%M%S")
            last = self._extract_last_frame(vp, run_id) if vp else ""
            seg_result = SegmentResult(
                segment_index=0, video_path=vp, last_frame_path=last,
                status="success", prompt_id=result.get("prompt_id", ""),
                duration_sec=0, overlap_prev=0, latent_path="")
            results_json = json.dumps(
                {"run_id": result.get("prompt_id", ""), "mode": plan.mode,
                 "backend": BACKEND_SCAIL2_NATIVE,
                 "segments": [seg_result.to_dict()]},
                ensure_ascii=False, indent=2)
            log = f"✅ 原生 SCAIL-2 一镜到底生成完成: {vp}"
            info("Runner-NativeSCAIL2", "完成: %s", vp)
            node_end("Runner-NativeSCAIL2", "OK")
            return (results_json, log, True)
        else:
            err = result.get("error", "unknown")
            error("Runner-NativeSCAIL2", "执行失败: %s", err)
            node_end("Runner-NativeSCAIL2", "FAIL")
            return ("", f"⚠ 原生 SCAIL-2 执行失败: {err}", False)

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
