"""原生 SCAIL-2 长视频节点驱动（预留适配器，默认不启用）。

为什么需要它：
FaboroHacks 工作流证明，原生 `SCAIL2ScheduledLongVideoWithSAM`（来自开源包
comfyui_scail2_multi_cond）即「一镜到底动作模仿」的现成实现——机制为
pose_video(驱动视频姿态) + reference 链式条件化(防漂移) + boundary_overlap/context_frames
边界重叠过渡，示例调度 599 帧≈37s，加段可到 1 分钟+ 且不劣化。

本 yunjii 引擎目前走「自分段 + 拼接」自有管线。原生节点在长视频(>30s)与「真·一镜到底
动作模仿」上更省心，故预留此适配器作为**后续切换入口**。

⚠️ 重要约束（2026-08-14 评估结论）：
- 本机当前**未安装** comfyui_scail2_multi_cond（沙箱无网、且需本机操作安装）。
- 沙箱**无 GPU / 无运行中的 ComfyUI**，无法在此实跑验证原生节点行为。
- 因此本适配器**绝不在导入期引用原生节点**（全部 lazy import），且默认 `SCAIL2_NATIVE_ENABLED=False`，
  现有自有管线不被替换、不受影响。真正切换前，必须在用户本机（RTX 3090）完成：
    ① pip/插件装 comfyui_scail2_multi_cond；
    ② 指向现有 SCAIL-2 权重（wan2.1_14B_SCAIL_2_fp8_scaled + lightx2v_I2V_14B 蒸馏 LoRA）；
    ③ 跑 FaboroHacks 参考工作流验证「1分钟+ 不劣化动作模仿」；
    ④ 把验证过的节点图接线写进 `build_native_graph()` 并置 `SCAIL2_NATIVE_ENABLED=True`。

本文件只提供：可用性探测 + 接线配方（基于已解析的 FaboroHacks 工作流），不含未经实跑的节点链接图。
"""

from __future__ import annotations

# 默认关闭：现有 yunjii 自有管线仍是主路径，原生节点切换需本机验证后再开。
SCAIL2_NATIVE_ENABLED = False

# 原生包里与「一镜到底动作模仿」相关的节点类名（来自 comfyui_scail2_multi_cond）。
NATIVE_NODE_CLASS = "SCAIL2ScheduledLongVideoWithSAM"
# 该包还提供更优的 Internal SAM 调度器变体 / 两阶段工作流——切换时优先评估。
NATIVE_INTERNAL_SAM_VARIANT = "SCAIL2InternalSAMScheduledLongVideoWithSAM"


def is_native_scail2_available() -> bool:
    """探测本机是否已安装原生 SCAIL-2 长视频节点。lazy import，导入本模块绝不会失败。"""
    try:
        import importlib
        # nodes 模块名以实际包为准；这里用包内标准节点注册入口。
        importlib.import_module("comfyui_scail2_multi_cond.nodes")
        return True
    except Exception:
        return False


def describe_native_scail2_wiring(plan=None) -> dict:
    """返回把 SegmentPlan 接进原生 SCAIL2ScheduledLongVideoWithSAM 的接线配方。

    仅描述输入/机制映射，不生成未经实跑的节点链接图（避免误导）。plan 可选，
    用来把时长/分段数换算成原生节点的调度表(boundary_overlap / context_frames)。
    """
    recipe = {
        "native_node": NATIVE_NODE_CLASS,
        "preferred_variant": NATIVE_INTERNAL_SAM_VARIANT,
        "inputs": {
            "pose_video": "驱动视频的姿态序列（动作模仿的核心输入）",
            "reference": "链式条件化锚点（上一段末帧/参考图），防长程漂移",
            "model": "wan2.1_14B_SCAIL_2_fp8_scaled（本机 fp8_scaled 权重，唯一可放下 24GB 的精度）",
            "distill_lora": "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16（4 步蒸馏，强制 steps=4）",
            "boundary_overlap": "段间重叠帧数（与猴子工作流对齐，潜空间 8 latent 帧 ≈ 32 像素帧）",
            "context_frames": "滑窗上下文帧数（单窗 81 帧，重叠 32 潜空间 fuse = 真·无缝）",
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


def build_native_graph(plan):
    """构造原生节点图（占位）。

    ⚠️ 未实现：该函数在 SCAIL2_NATIVE_ENABLED=True 且本机已装包、并经 GPU 实跑验证前不得被调用。
    届时依据 describe_native_scail2_wiring() 的配方，把节点/链接写在此处（参考 FaboroHacks 工作流）。
    """
    raise NotImplementedError(
        "原生 SCAIL-2 节点图尚未实跑验证。请先在本机安装 comfyui_scail2_multi_cond 并跑通 "
        "FaboroHacks 参考工作流，再把验证过的接线写进 build_native_graph()，最后置 "
        "SCAIL2_NATIVE_ENABLED=True。当前 yunjii 引擎继续走自有管线。"
    )
