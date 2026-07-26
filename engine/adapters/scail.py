"""
SCAIL-2 生成后端适配器（骨架 / 接口验证版）。

设计：直接继承 DirectAdapter，复用其内联执行核心
（init_executor / execute_inline / cleanup_executor / _get_output_node_ids /
_extract_video_from_history 等），只重写：
  - discover_nodes        : 识别 SCAIL-2 原生节点（WanSCAILToVideo 等）
  - modify_workflow_for_segment : 把 planner 的 SegmentInfo 映射到 SCAIL-2 工作流

SCAIL-2 与现有 WanVideo 骨骼路线范式不同（无骨架、端到端），节点角色也不同，
因此用独立的 SCAILNodeMap 描述节点映射，不污染 WanVideo 的 NodeMap。

注意：
  - SCAIL-2 自带长视频分块（Base/Extend 子图，每段 ~76 帧 + 5 帧重叠，
    靠 WanSCAILToVideo 的 previous_frames 串联）。本适配器按 planner 的
    SegmentInfo（start_frame/end_frame/target_frames/ref_strategy）驱动它：
      * 首段（ref_strategy=user_image）：Animation 模式，参考图 = 角色参考图
      * 后续段（ref_strategy=prev_last_frame）：把前段末帧作为 previous_frames 串联
  - 真实接模型前，workflows/scail2_template.json 应替换为官方 SCAIL-2 工作流，
    并确保 COMFYUI 已更新到含 WanSCAILToVideo / SCAIL2ColoredMask 的版本。
"""

import json
import os
import time

from .direct import DirectAdapter
from ..types import REF_STRATEGY_PREV_LAST_FRAME
from ..debug_log import info, warn, error as log_error


# ---- SCAIL-2 原生节点 class_type（来自 docs.comfy.org SCAIL-2 教程）----
# 若你安装的节点包名称不同，改这里即可，无需动逻辑。
SCAIL_CORE = "WanSCAILToVideo"          # 核心：参考图 + 驱动视频 + 掩码 -> 视频
SCAIL_MASK = "SCAIL2ColoredMask"        # 彩色掩码：合并 SAM3 跟踪为驱动/参考两路掩码
SCAIL_SAM3_TRACK = "SAM3_VideoTrack"     # SAM3 跟踪（驱动视频 / 参考图 各一个）
SCAIL_VIDEO_LOAD = "VHS_LoadVideo"      # 驱动视频加载
SCAIL_VIDEO_COMBINE = "VHS_VideoCombine"  # 输出视频保存
SCAIL_REF_LOAD = "LoadImage"            # 角色参考图加载


class SCAILNodeMap:
    """SCAIL-2 工作流的节点映射（与 WanVideo 的 NodeMap 字段不同）。"""

    def __init__(self):
        self.core = ""          # WanSCAILToVideo 节点 id
        self.mask = ""          # SCAIL2ColoredMask 节点 id
        self.sam3_video = ""    # SAM3_VideoTrack（驱动视频）节点 id
        self.sam3_image = ""    # SAM3_VideoTrack（参考图）节点 id
        self.video_combine = ""  # VHS_VideoCombine 节点 id
        self.ref_image = ""     # LoadImage（角色参考图）节点 id
        self.driving_video = ""  # VHS_LoadVideo 节点 id

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v}

    def is_valid(self):
        # 核心 + 输出 + 驱动视频 + 角色参考图 四者必备
        return bool(self.core and self.video_combine and self.driving_video and self.ref_image)


class SCAILAdapter(DirectAdapter):
    """SCAIL-2 生成后端。复用 DirectAdapter 的执行核心，重写节点发现与分段改写。"""

    def discover_nodes(self, workflow):
        nm = SCAILNodeMap()
        items = (
            workflow.get("nodes", {}).items()
            if isinstance(workflow.get("nodes"), dict)
            else workflow.items()
        )
        for nid, ndata in items:
            if not isinstance(ndata, dict):
                continue
            ct = ndata.get("class_type", "")
            if ct == SCAIL_CORE:
                nm.core = nid
            elif ct == SCAIL_MASK:
                nm.mask = nid
            elif ct == SCAIL_SAM3_TRACK:
                # 两个 SAM3_VideoTrack（驱动视频 / 参考图）：先出现的是驱动视频
                if not nm.sam3_video:
                    nm.sam3_video = nid
                else:
                    nm.sam3_image = nid
            elif ct == SCAIL_VIDEO_COMBINE:
                nm.video_combine = nid
            elif ct == SCAIL_REF_LOAD:
                if not nm.ref_image:
                    nm.ref_image = nid
            elif ct == SCAIL_VIDEO_LOAD:
                nm.driving_video = nid
        if not nm.is_valid():
            miss = []
            if not nm.core:
                miss.append(SCAIL_CORE)
            if not nm.video_combine:
                miss.append(SCAIL_VIDEO_COMBINE)
            if not nm.driving_video:
                miss.append(SCAIL_VIDEO_LOAD)
            if not nm.ref_image:
                miss.append(SCAIL_REF_LOAD)
            warn("SCAILAdapter", "工作流缺少必要 SCAIL-2 节点: %s", ", ".join(miss))
        return nm

    def modify_workflow_for_segment(self, workflow, node_map, seg, ref_image_path, pose_dir="", run_id="", user_ref_path=""):
        """
        把一个 SegmentInfo 映射到 SCAIL-2 工作流输入。

        SCAIL-2 与 WanVideo 的关键区别：角色身份来自「参考图」，动作连贯来自
        「previous_frames」。两者是不同输入，因此：
          - 角色参考图（user_ref_path）：每段都注入，保证身份一致（不随段变化）。
          - previous_frames（ref_image_path=前段末帧）：仅后续段用于动作连贯串联。
          - 驱动视频：用 VHS_LoadVideo.skip_first_frames = seg.start_frame 偏移取本段动作。
          - 核心节点：segment_index(1-based) / frame_count / width / height / prompt / replace_mode。
        """
        wf = json.loads(json.dumps(workflow))

        # 1) 角色参考图（身份来源）：每段都注入用户参考图，避免后续段误用前段末帧当作新身份
        char_ref = user_ref_path or ref_image_path
        if char_ref and node_map.ref_image and node_map.ref_image in wf:
            img_name = self._copy_to_input(char_ref)
            if img_name:
                wf[node_map.ref_image]["inputs"]["image"] = img_name
                info("SCAILAdapter", "角色参考图注入: node=%s, image=%s", node_map.ref_image, img_name)

        # 2) 连续帧（动作连贯）：后续段把前段末帧作为 previous_frames 串联核心节点
        if seg.ref_strategy == REF_STRATEGY_PREV_LAST_FRAME and ref_image_path and node_map.core and node_map.core in wf:
            prev_name = self._copy_to_input(ref_image_path)
            prev_node = "yunjii_scail_prev_frame"
            wf[prev_node] = {"class_type": "LoadImage", "inputs": {"image": prev_name}}
            wf[node_map.core].setdefault("inputs", {})["previous_frames"] = [prev_node, 0]
            info("SCAILAdapter", "previous_frames 串联: node=%s, image=%s", node_map.core, prev_name)

        # 3) 驱动视频起始帧偏移
        if node_map.driving_video and node_map.driving_video in wf:
            di = wf[node_map.driving_video].setdefault("inputs", {})
            di["skip_first_frames"] = max(0, seg.start_frame)
            info("SCAILAdapter", "驱动视频起始帧偏移 -> %d", seg.start_frame)

        # 4) 核心节点参数
        if node_map.core and node_map.core in wf:
            ci = wf[node_map.core].setdefault("inputs", {})
            ci["segment_index"] = seg.index + 1
            ci["frame_count"] = seg.target_frames
            ci["width"] = seg.params.get("width", 832)
            ci["height"] = seg.params.get("height", 480)
            if seg.prompt:
                ci["prompt"] = seg.prompt
            # 本 V2V 链路驱动同一角色：统一用 Animation 模式（false）；
            # Replacement 模式（true）适用于"替换驱动视频里某个人"的场景，按需改。
            ci["replace_mode"] = False
            info("SCAILAdapter", "核心节点配置: segment_index=%d, frame_count=%d, replace_mode=%s",
                 ci["segment_index"], ci["frame_count"], ci.get("replace_mode"))

        # 5) 输出文件名前缀（便于在 output/yunjii_scail/<run_id>/ 下区分分段）
        if node_map.video_combine and node_map.video_combine in wf:
            vc = wf[node_map.video_combine].setdefault("inputs", {})
            if "filename_prefix" in vc:
                sub = run_id or time.strftime("%Y%m%d_%H%M%S")
                vc["filename_prefix"] = f"yunjii_scail/{sub}/seg{seg.index}"

        return wf
