"""原生 SCAIL-2 长视频节点驱动适配器（默认不启用，源码级准确接线）。

为什么需要它：
FaboroHacks 工作流证明，原生 `comfyui_scail2_multi_cond` 包里的长视频调度节点即
「一镜到底动作模仿」的现成实现。机制为：
  pose_video(驱动视频姿态) + reference 链式条件化(防漂移) + boundary_overlap 边界重叠过渡。
示例调度 599 帧 ≈ 37s，追加调度段可到 1 分钟+ 且不劣化。

本 yunjii 引擎目前走「自分段 + 拼接」自有管线。原生节点在长视频(>30s)与「真·一镜到底
动作模仿」上更省心，故预留此适配器作为**后续切换入口**。

⚠️ 重要约束（2026-08-14 评估结论 + 2026-08-15 源码核对）：
- 本机当前**未安装** comfyui_scail2_multi_cond（沙箱无网、且需本机操作安装）。
- 沙箱**无 GPU / 无运行中的 ComfyUI**，无法在此实跑验证原生节点行为。
- 因此本适配器**绝不在导入期引用原生节点**（全部 lazy import），且默认 `SCAIL2_NATIVE_ENABLED=False`，
  现有自有管线不被替换、不受影响。真正切换前，必须在用户本机（RTX 3090）完成：
    ① pip/插件装 comfyui_scail2_multi_cond（git clone 后 restart ComfyUI）；
    ② 指向现有 SCAIL-2 权重（wan2.1_14B_SCAIL_2_fp8_scaled + lightx2v_I2V_14B 蒸馏 LoRA）；
    ③ 跑 FaboroHacks 参考工作流验证「1 分钟+ 不劣化动作模仿」；
    ④ 把验证过的节点图接线写进 `build_native_graph()` 并置 `SCAIL2_NATIVE_ENABLED=True`。

类名与端口均来自 2026-08-15 对仓库 nodes.py 的源码核对（raw.githubusercontent 抓取）：
- `SCAIL2SegmentPlanBuilder`  INPUT: segment_count(INT) + 动态 segment_{i}_frames/reference/prompt/
  negative/boundary_overlap  OUTPUT: ("segment_plan"(STRING), "summary"(STRING))
- `SCAIL2ScheduledLongVideoWithSAM`(FaboroHacks 实际使用的生成器, SCAIL2ScheduledLongVideo 子类, 自动出 SAM 遮罩)
  REQUIRED: 与基类同(model/clip/vae/sampler/sigmas/clip_vision/pose_video/segment_plan/seed/cfg/mode/
  max_frames/max_chunk_frames/overlap_frames/reference_count/color_correction) + 专有必填
  object_indices(STRING)/reference_object_indices(STRING)/sort_by(["none","left_to_right","area"])/
  sam_detection_threshold(FLOAT)/sam_max_objects(INT)/sam_detect_interval(INT)；OPTIONAL: sam_model(MODEL)/
  sam_conditioning(CONDITIONING)/reference_{i}(IMAGE)；**无** pose_video_mask(由 SAM 自动生成)。
  OUTPUT: ("frames"(IMAGE), "used_pose_video_mask", "used_reference_mask_timeline", "summary")
- `SCAIL2ScheduledLongVideo`(基类, 外部 mask 版)：与 WithSAM 同构但需手绘 pose_video_mask / reference_{i}_mask。

loader(model/clip/vae/sampler/sigmas/clip_vision) 不在本适配器臆测：由 `ctx` 注入已构建的
loader 节点引用，原生生成链路本身用上列真实端口名写死。
"""

from __future__ import annotations

# 默认关闭：现有 yunjii 自有管线仍是主路径，原生节点切换需本机验证后再开。
SCAIL2_NATIVE_ENABLED = False

# 原生包里与「一镜到底动作模仿」相关的节点类名（2026-08-15 安装后源码核对，确证）。
NATIVE_NODE_CLASS = "SCAIL2ScheduledLongVideoWithSAM"   # FaboroHacks 实际使用的生成器(自动出 SAM 遮罩, SCAIL2ScheduledLongVideo 子类)
NATIVE_BASE_NODE_CLASS = "SCAIL2ScheduledLongVideo"     # 基类(外部 mask 版, 需手绘 pose_video_mask / reference_{i}_mask)
NATIVE_PLAN_BUILDER_CLASS = "SCAIL2SegmentPlanBuilder"  # 由分段参数生成 segment_plan 字符串(RETURN_NAMES=segment_plan)

# 包内常量(依据源码 MAX_REFERENCES=8, 段数/参考数上限)
MAX_REFERENCES = 8
# 单块帧数对齐 4n+1 且上限 81（与猴子工作流/原生节点 max_chunk_frames 一致）
CHUNK_FRAMES_DEFAULT = 81
OVERLAP_FRAMES_DEFAULT = 5


def is_native_scail2_available() -> bool:
    """探测本机是否已安装原生 SCAIL-2 长视频节点。lazy import，导入本模块绝不会失败。"""
    try:
        import importlib
        importlib.import_module("comfyui_scail2_multi_cond.nodes")
        return True
    except Exception:
        return False


def _seg_link(ref):
    """把 ctx 里的 loader/资源引用规整为 ComfyUI 链接元组 [node_id, output_index]。"""
    if isinstance(ref, (list, tuple)) and len(ref) == 2:
        return [str(ref[0]), int(ref[1])]
    # 允许直接传 node_id 字符串（默认取第 0 输出）
    return [str(ref), 0]


def describe_native_scail2_wiring(plan=None) -> dict:
    """返回把 yunjii SegmentPlan 接进原生 SCAIL-2 长视频节点的接线配方（源码级端口名）。"""
    recipe = {
        "native_node": NATIVE_NODE_CLASS,
        "base_node": NATIVE_BASE_NODE_CLASS,
        "plan_builder": NATIVE_PLAN_BUILDER_CLASS,
        "inputs": {
            "pose_video": "驱动视频的姿态序列(IMAGE)——动作模仿的核心输入，来自 ctx['pose_video']",
            "reference_i": "分段参考图(IMAGE)，链式条件化锚点(防长程漂移)，来自 ctx['references']",
            "reference_i_mask": "参考图遮罩(IMAGE)，可选，来自 ctx['reference_masks']",
            "segment_plan": "由 SCAIL2SegmentPlanBuilder 生成的调度字符串(STRING)，定义各段帧数/参考/提示/边界重叠",
            "model/clip/vae/sampler/sigmas/clip_vision": "由 ctx 注入的 loader 节点引用(Wan 系列加载器)",
            "max_chunk_frames": "单块最大帧数(17~81, 对齐 4n+1)，默认 81",
            "overlap_frames": "块间重叠帧数(边界过渡)，默认 5；与段间 boundary_overlap 对应",
            "mode": "['replacement','animation']，默认 replacement",
            "reference_count": "参与条件化的参考图数量(≤8)",
            "color_correction": "BOOLEAN，默认 True(色彩一致性)",
            "cache_mode": "['disk','off']，默认 disk(缓存加速重算)",
        },
        "continuity_mechanism": (
            "pose_video 驱动 + reference 链式 + 边界重叠 → 整片在同一条去噪轨迹连续，"
            "等价于本引擎 B 方案的真·无缝，但由原生节点原生实现、更省心"
        ),
        "duration_estimate": "示例调度 599 帧 ≈ 37s；追加调度段可到 1 分钟+ 且不劣化",
        "notes": [
            "fp8 在 sm_86(3090) 走 torch._scaled_mm 软回退，与 expandable_segments(VMM) 可能竞争；"
            "切换时启动前设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False",
            "蒸馏 LoRA 检测必须覆盖所有输入值含 'distill'，否则 steps 不匹配会结构性崩坏",
            "Internal SAM 变体类名依截断推断，本机切换前需在 nodes.py 再核对一次真实类名",
        ],
    }
    if plan is not None:
        try:
            total = int(getattr(plan, "total_frames", 0) or 0)
            recipe["schedule_hint_frames"] = total
            recipe["schedule_hint_seconds"] = round(total / 16.0, 1) if total else 0.0
        except Exception:
            pass
    return recipe


def _builder_inputs_from_plan(plan) -> dict:
    """把 yunjii SegmentPlan 翻译成 SCAIL2SegmentPlanBuilder 的 INPUT(动态段字段)。

    用 Builder 节点生成 segment_plan 字符串，避免手写未经证实的 plan 文本格式。
    段字段逐个映射：frames←段帧数, reference←该段参考图序号, prompt/negative←段提示词,
    boundary_overlap←段间重叠。段数超过 MAX_REFERENCES(8) 时截断并告警。
    """
    segs = getattr(plan, "segments", None) or []
    n = min(len(segs), MAX_REFERENCES)
    if len(segs) > MAX_REFERENCES:
        import logging
        logging.getLogger("yunjii").warning(
            "[scail2_native] 段数 %d 超过原生节点上限 %d，已截断为前 %d 段",
            len(segs), MAX_REFERENCES, MAX_REFERENCES,
        )
    inputs = {"segment_count": (n, {"min": 1, "max": MAX_REFERENCES, "step": 1})}
    for i in range(1, n + 1):
        s = segs[i - 1]
        # 段帧数：优先显式帧数，否则用段内帧计数
        frames = int(getattr(s, "frames", 0) or getattr(s, "frame_count", 0) or 0)
        # 该段参考图序号：取 1..min(i, MAX_REFERENCES)（首段用第 1 张参考，后续链式递增）
        ref_idx = min(i, MAX_REFERENCES)
        prompt = str(getattr(s, "prompt", "") or "")
        negative = str(getattr(s, "negative_prompt", "") or "")
        overlap = int(getattr(s, "boundary_overlap", OVERLAP_FRAMES_DEFAULT) or OVERLAP_FRAMES_DEFAULT)
        inputs[f"segment_{i}_frames"] = (frames, {"min": 1, "max": 100000, "step": 1})
        inputs[f"segment_{i}_reference"] = (ref_idx, {"min": 1, "max": MAX_REFERENCES, "step": 1})
        inputs[f"segment_{i}_prompt"] = (prompt, {"multiline": True})
        inputs[f"segment_{i}_negative"] = (negative, {"multiline": True})
        inputs[f"segment_{i}_boundary_overlap"] = (overlap, {"min": -1, "max": 33, "step": 1})
    return inputs


def build_native_graph(plan, ctx):
    """构造原生 SCAIL-2 长视频节点图（ComfyUI prompt dict）。

    接线（源码级准确，类名/端口来自 2026-08-15 安装后核对）：
      ① SCAIL2SegmentPlanBuilder  ← 由 plan 翻译的分段参数  →  segment_plan(STRING)
      ② SCAIL2ScheduledLongVideoWithSAM  ← loaders(ctx) + pose_video(ctx)
                                     + segment_plan[①,0] + reference_i(ctx) + SAM 专有必填项
                                     →  frames(IMAGE) 等（SAM 自动出遮罩，无需 pose_video_mask）

    ⚠️ 未启用前不得被调用：SCAIL2_NATIVE_ENABLED 必须为 True，且本机已装包、并经 GPU 实跑验证。
    ctx 必须提供：
      model/clip/vae/sampler/sigmas/clip_vision : loader 节点引用(node_id 或 [node_id, idx])
      pose_video                                : 驱动视频帧(IMAGE) 节点引用
      references                                : 参考图列表(每个为节点引用)
    可选 ctx：seed/cfg/mode/max_frames/max_chunk_frames/overlap_frames/color_correction/cache_mode/
      object_indices/reference_object_indices/sort_by/sam_detection_threshold/sam_max_objects/
      sam_detect_interval/sam_model/sam_conditioning
    返回标准 ComfyUI prompt dict（含 nodes 与链接元组）。
    """
    if not SCAIL2_NATIVE_ENABLED:
        raise RuntimeError(
            "SCAIL2_NATIVE_ENABLED=False：原生节点图未启用。请先在本机安装 comfyui_scail2_multi_cond、"
            "跑通 FaboroHacks 验证后，再置 SCAIL2_NATIVE_ENABLED=True。"
        )
    if not isinstance(ctx, dict):
        raise ValueError("build_native_graph: ctx 必须提供 loader/资源节点引用字典")

    required_keys = ("model", "clip", "vae", "sampler", "sigmas", "clip_vision", "pose_video", "references")
    missing = [k for k in required_keys if k not in ctx]
    if missing:
        raise ValueError(f"build_native_graph: ctx 缺少必要键: {missing}")

    references = ctx.get("references") or []
    ref_count = max(len(references), 1)
    ref_count = min(ref_count, MAX_REFERENCES)

    # ① 分段计划构建节点
    builder_id = "scail2_plan_builder"
    prompt = {
        builder_id: {
            "class_type": NATIVE_PLAN_BUILDER_CLASS,
            "inputs": _builder_inputs_from_plan(plan),
        },
        # ② 主生成器（FaboroHacks 使用的 SCAIL2ScheduledLongVideoWithSAM，自动出 SAM 遮罩）
        "scail2_scheduled_long_video": {
            "class_type": NATIVE_NODE_CLASS,
            "inputs": {
                "model": _seg_link(ctx["model"]),
                "clip": _seg_link(ctx["clip"]),
                "vae": _seg_link(ctx["vae"]),
                "sampler": _seg_link(ctx["sampler"]),
                "sigmas": _seg_link(ctx["sigmas"]),
                "clip_vision": _seg_link(ctx["clip_vision"]),
                "pose_video": _seg_link(ctx["pose_video"]),
                "segment_plan": [builder_id, 0],
                "seed": int(ctx.get("seed", 1) or 1),
                "cfg": float(ctx.get("cfg", 1.0) or 1.0),
                "mode": ctx.get("mode", "replacement"),
                "max_frames": int(ctx.get("max_frames", 0) or 0),
                "max_chunk_frames": int(ctx.get("max_chunk_frames", CHUNK_FRAMES_DEFAULT) or CHUNK_FRAMES_DEFAULT),
                "overlap_frames": int(ctx.get("overlap_frames", OVERLAP_FRAMES_DEFAULT) or OVERLAP_FRAMES_DEFAULT),
                "reference_count": ref_count,
                "color_correction": bool(ctx.get("color_correction", True)),
                # —— SCAIL2ScheduledLongVideoWithSAM 专有必填项（默认空/推荐值）——
                "object_indices": str(ctx.get("object_indices", "") or ""),
                "reference_object_indices": str(ctx.get("reference_object_indices", "") or ""),
                "sort_by": ctx.get("sort_by", "left_to_right"),
                "sam_detection_threshold": float(ctx.get("sam_detection_threshold", 0.5) or 0.5),
                "sam_max_objects": int(ctx.get("sam_max_objects", 2) or 2),
                "sam_detect_interval": int(ctx.get("sam_detect_interval", 2) or 2),
                "cache_mode": ctx.get("cache_mode", "disk"),
                # optional：SAM 模型（留空则由节点内部处理）
                "sam_model": _seg_link(ctx["sam_model"]) if ctx.get("sam_model") else None,
                "sam_conditioning": _seg_link(ctx["sam_conditioning"]) if ctx.get("sam_conditioning") else None,
            },
        },
    }
    # 注入参考图（reference_1..N，WithSAM 的 optional 端口）
    gen_inputs = prompt["scail2_scheduled_long_video"]["inputs"]
    for i in range(1, ref_count + 1):
        gen_inputs[f"reference_{i}"] = _seg_link(references[i - 1])
    return prompt


# —— 向后兼容：早期草案的占位函数名，避免旧调用方 ImportError ——
def build_native_graph_legacy(plan):  # pragma: no cover - 兼容垫片
    raise NotImplementedError(
        "原生 SCAIL-2 节点图尚未实跑验证。请先在本机安装 comfyui_scail2_multi_cond 并跑通 "
        "FaboroHacks 参考工作流，再把验证过的接线写进 build_native_graph(plan, ctx)，最后置 "
        "SCAIL2_NATIVE_ENABLED=True。当前 yunjii 引擎继续走自有管线。"
    )
