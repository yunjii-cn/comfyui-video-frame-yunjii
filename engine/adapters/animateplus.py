"""
Tier 2 暖启动适配器（AnimatePlus SCAIL-2 路线）。

设计目标：把 SCAIL-2 生成从 WanVideoWrapper 家族重包进 WanAnimatePlus 家族
（参考「分段队列」丝滑工作流的做法），用其原生时序建模 + 多段重叠融合得到
「连续 + 画质」兼得的真·一镜到底；并对 seg>0 把上段真实成片末帧注入
`WanAnimatePlus SCAIL_2 Embeds.prefix_frames`，做跨段暖启动（加分项）。

复用：
- DirectAdapter 的内联执行核心（execute_inline / init_executor / cleanup_executor /
  _copy_to_input / _extract_last_frame）。
- SCAILAdapter 的静态预处理（Set/Get 重连、Reroute 解析、未注册节点清理、
  UI→API 转换、模型名模糊匹配、pose_images 回退）——这些与具体家族无关，通用。

非破坏性：若模板不是 WanAnimatePlus SCAIL_2 工作流（缺关键节点），discover 失败，
runner 会优雅回退到标准 SCAIL 路线；prefix 注入任何异常都被 try/except 吞掉，
最坏情况只是「无 prefix 的 WanAnimatePlus 多段」（仍比纯 WanVideoWrapper 连续）。

注意：本适配器端到端跑通依赖本机 WanAnimatePlus 节点集 + 权重 + 显存，需用户在
GPU 上验证（我方无 GPU）。若报错，回退「多段无缝」或「单遍连贯(方案C)」即可，
现有 WanVideoWrapper SCAIL 路线不受影响。
"""

import os
import json
import cv2

from .direct import DirectAdapter, NodeMap
from .scail import SCAILAdapter, DEFAULT_NEGATIVE
from ..debug_log import info, warn, error as log_error

# WanAnimatePlus 家族关键 class_type
AP_SCAIL_EMBEDS = "WanAnimatePlus SCAIL_2 Embeds"
AP_SAMPLER_FROM_SETTINGS = "WanAnimatePlus SamplerFromSettings"
AP_SAMPLER = "WanAnimatePlus Sampler"
AP_TEXT_ENCODE = "WanVideoTextEncodeCached"
AP_VHS_LOADVIDEO = "VHS_LoadVideo"
AP_REF_IMAGE = "LoadImage"
AP_VIDEO_COMBINE = "VHS_VideoCombine"
AP_IMAGEFROMBATCH = "ImageFromBatch"
AP_CONTEXT_OPTIONS = "WanAnimatePlus ContextOptions"

# prefix_frames 最多注入帧数（WanAnimatePlus 限制 ≤5，最多 17 帧 prefix）
PREFIX_MAX_FRAMES = 5


class AnimatePlusNodeMap(NodeMap):
    def __init__(self):
        super().__init__()
        self.driving_video = ""
        self.ap_sampler = ""
        self.context_options = ""

    def to_dict(self):
        d = super().to_dict()
        d.update({"ap_sampler": self.ap_sampler, "context_options": self.context_options})
        return d

    def is_valid(self):
        return bool(self.animate_embeds and self.video_combine and self.ref_image and self.text_encode)


class AnimatePlusSCAILAdapter(DirectAdapter):
    """Tier 2 暖启动：WanAnimatePlus SCAIL_2 家族 + prefix_frames 跨段暖启动。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.driving_video_path = ""
        self.model_precision = "fp8"
        self.prev_video_path = ""  # 上一段成片视频路径（用于 prefix 注入）

    # ------------------------------------------------------------------
    # 节点发现
    # ------------------------------------------------------------------
    def discover_nodes(self, workflow):
        nm = AnimatePlusNodeMap()
        vhs_candidates = []
        for nid, ndata in DirectAdapter._iter_nodes(workflow):
            if not isinstance(ndata, dict):
                continue
            ct = ndata.get("class_type") or ndata.get("type", "")
            if ct == AP_SCAIL_EMBEDS and not nm.animate_embeds:
                nm.animate_embeds = nid
            elif ct in (AP_SAMPLER_FROM_SETTINGS, AP_SAMPLER) and not nm.ap_sampler:
                nm.ap_sampler = nid
                nm.sampler = nid
            elif ct == AP_TEXT_ENCODE and not nm.text_encode:
                nm.text_encode = nid
            elif ct == AP_VHS_LOADVIDEO and not nm.driving_video:
                nm.driving_video = nid
            elif ct == AP_VIDEO_COMBINE:
                prefix, save_out = DirectAdapter._vhs_meta(ndata)
                vhs_candidates.append((nid, prefix, save_out))
            elif ct == AP_CONTEXT_OPTIONS and not nm.context_options:
                nm.context_options = nid

        # 参考图：优先从 animate_embeds.ref_image 上游回溯到真正的 LoadImage
        # （参考工作流里 ref 常经 Reroute/ImageResize 才连到 LoadImage）。
        if nm.animate_embeds and nm.animate_embeds in workflow:
            ref_link = workflow[nm.animate_embeds].get("inputs", {}).get("ref_image")
            if isinstance(ref_link, list) and len(ref_link) >= 1:
                li = self._find_upstream_loadimage(workflow, ref_link[0])
                if li:
                    nm.ref_image = li
        if not nm.ref_image:
            # 退化：title 含 参考/reference/人物 的 LoadImage（兼容中文标题）
            for nid, ndata in DirectAdapter._iter_nodes(workflow):
                if isinstance(ndata, dict) and ndata.get("class_type") == AP_REF_IMAGE:
                    title = (ndata.get("title") or ndata.get("_meta", {}).get("title") or "")
                    if any(k in title.lower() for k in ("参考", "reference", "ref", "人物")):
                        nm.ref_image = nid
                        break
        if not nm.ref_image:
            # 再退化：第一个 LoadImage
            for nid, ndata in DirectAdapter._iter_nodes(workflow):
                if isinstance(ndata, dict) and ndata.get("class_type") == AP_REF_IMAGE:
                    nm.ref_image = nid
                    break

        nm.video_combine = DirectAdapter._select_primary_vhs(vhs_candidates)
        if not nm.is_valid():
            miss = [n for n, v in [
                ("WanAnimatePlus SCAIL_2 Embeds", nm.animate_embeds),
                ("VHS_VideoCombine", nm.video_combine),
                ("LoadImage", nm.ref_image),
                ("WanVideoTextEncodeCached", nm.text_encode),
            ] if not v]
            warn("AnimatePlusAdapter", "工作流缺少必要 WanAnimatePlus 节点: %s", ", ".join(miss))
        return nm

    def _find_upstream_loadimage(self, workflow, start_node_id, max_depth=10):
        """从某节点沿 link([src_node, slot]) 向上回溯，返回第一个 LoadImage 的 id。"""
        seen = set()
        stack = [start_node_id]
        while stack and len(seen) < max_depth + 1:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = workflow.get(nid)
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == AP_REF_IMAGE:
                return nid
            for v in node.get("inputs", {}).values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    stack.append(v[0])
        return ""

    # ------------------------------------------------------------------
    # 预处理（复用 SCAILAdapter 静态手术）
    # ------------------------------------------------------------------
    def prepare_workflow(self, workflow_raw):
        if not isinstance(workflow_raw, dict):
            warn("AnimatePlusAdapter", "prepare_workflow: 输入非 dict")
            return {}
        first = next(iter(workflow_raw), None)
        if first is not None and isinstance(workflow_raw[first], dict) \
                and ("class_type" in workflow_raw[first] or "type" in workflow_raw[first]):
            api = SCAILAdapter._prune_api_missing(dict(workflow_raw))
            SCAILAdapter._fix_model_names(api)
            # _fix_pose_images 作用于 WanVideoAddSCAILPoseEmbeds，本家族无该节点→安全 no-op
            return api
        full = json.loads(json.dumps(workflow_raw))
        full = SCAILAdapter._rewire_setget(full)
        full = SCAILAdapter._resolve_reroutes(full)
        full = SCAILAdapter._delete_unregistered(full)
        full = SCAILAdapter._delete_dangling_vhs(full)
        full = SCAILAdapter._keep_main_chain(full)
        api = SCAILAdapter._convert_full_to_api(full, SCAILAdapter._node_class_mappings())
        SCAILAdapter._fix_model_names(api)
        info("AnimatePlusAdapter", "prepare_workflow: 整理后 %d 个节点", len(api))
        return api

    # ------------------------------------------------------------------
    # 段参数注入
    # ------------------------------------------------------------------
    def modify_workflow_for_segment(self, workflow, node_map, seg, ref_image_path,
                                    pose_dir="", run_id="", user_ref_path="",
                                    prev_video_path=""):
        wf = json.loads(json.dumps(workflow))
        char_ref = user_ref_path or ref_image_path

        # 1) 参考图（每段注入，保证身份一致）
        if char_ref and node_map.ref_image and node_map.ref_image in wf:
            img_name = self._copy_to_input(char_ref)
            if img_name:
                wf[node_map.ref_image].setdefault("inputs", {})["image"] = img_name
                info("AnimatePlusAdapter", "参考图注入: node=%s, image=%s", node_map.ref_image, img_name)

        # 2) WanAnimatePlus SCAIL_2 Embeds：分辨率 + 帧数（4k+1 对齐）
        if node_map.animate_embeds and node_map.animate_embeds in wf:
            ae = wf[node_map.animate_embeds]
            ae.setdefault("inputs", {})
            w = seg.params.get("width", 832) if isinstance(seg.params, dict) else 832
            h = seg.params.get("height", 480) if isinstance(seg.params, dict) else 480
            n = seg.target_frames
            aligned = max(9, ((n - 1) // 4) * 4 + 1)
            ae["inputs"]["width"] = w
            ae["inputs"]["height"] = h
            ae["inputs"]["num_frames"] = aligned
            if "frame_window_size" in ae["inputs"]:
                ae["inputs"]["frame_window_size"] = aligned
            info("AnimatePlusAdapter", "Embeds: width=%d height=%d num_frames=%d(aligned %d)", w, h, n, aligned)

        # 3) 驱动视频（动作源）+ 分段偏移 + 段长
        if node_map.driving_video and node_map.driving_video in wf and self.driving_video_path \
                and os.path.isfile(self.driving_video_path):
            fname = self._copy_to_input(self.driving_video_path)
            di = wf[node_map.driving_video].setdefault("inputs", {})
            if fname and "video" in di:
                di["video"] = fname
            di["skip_first_frames"] = max(0, seg.start_frame)
            di["frame_load_cap"] = seg.target_frames
            if "select_every_nth" in di:
                di["select_every_nth"] = 1
            info("AnimatePlusAdapter", "驱动视频: %s, 偏移=%d, 段长=%d", fname, seg.start_frame, seg.target_frames)
        elif node_map.driving_video:
            warn("AnimatePlusAdapter", "未设置驱动视频路径，WanAnimatePlus SCAIL-2 无法生成动作")

        # 4) 提示词
        if node_map.text_encode and node_map.text_encode in wf:
            self._set(wf, node_map.text_encode, "positive_prompt", seg.prompt or "")
            self._set(wf, node_map.text_encode, "negative_prompt",
                      (seg.params.get("negative") if isinstance(seg.params, dict) else None)
                      or DEFAULT_NEGATIVE)

        # 5) prefix_frames 暖启动（seg>0）：上段成片末帧 → 硬冻结到本段 latent 开头
        if seg.index > 0 and prev_video_path and os.path.isfile(prev_video_path):
            self._inject_prefix(wf, node_map, prev_video_path)

        return wf

    # ------------------------------------------------------------------
    # prefix 注入（best-effort，任何异常都吞掉，保证工作流仍可跑）
    # ------------------------------------------------------------------
    def _inject_prefix(self, wf, node_map, prev_video_path):
        try:
            if not node_map.animate_embeds or node_map.animate_embeds not in wf:
                return
            ae = wf[node_map.animate_embeds]
            if "prefix_frames" not in ae.get("inputs", {}):
                return  # 该节点版本不支持 prefix
            if ae["inputs"].get("prefix_frames") is not None:
                return  # 已被模板占用，不覆盖

            # 复制上段成片到 input 目录（与驱动视频同机制），用 VHS 加载其尾部帧
            fname = self._copy_to_input(prev_video_path)
            if not fname:
                warn("AnimatePlusAdapter", "prefix: 复制上段成片失败，跳过 prefix 注入")
                return

            total = self._video_frame_count(prev_video_path)
            n = PREFIX_MAX_FRAMES if total is None else min(PREFIX_MAX_FRAMES, total)
            if n <= 0:
                warn("AnimatePlusAdapter", "prefix: 上段成片无有效帧，跳过 prefix 注入")
                return

            vhs_id = "yunjii_prev_vhs"
            ifb_id = "yunjii_prev_ifb"
            wf[vhs_id] = {
                "class_type": AP_VHS_LOADVIDEO,
                "inputs": {
                    "video": fname,
                    "force_rate": 0,
                    "frame_load_cap": 0,   # 0=全加载，由 ImageFromBatch 抽尾帧
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
            }
            wf[ifb_id] = {
                "class_type": AP_IMAGEFROMBATCH,
                "inputs": {
                    "image": [vhs_id, 0],
                    "batch_index": -n,
                    "length": n,
                },
            }
            ae["inputs"]["prefix_frames"] = [ifb_id, 0]
            info("AnimatePlusAdapter", "prefix_frames 注入: 上段成片%s → 末%d帧暖启动", prev_video_path, n)
        except Exception as e:
            warn("AnimatePlusAdapter", "prefix_frames 注入异常(已跳过，不影响主生成): %s", str(e)[:200])

    @staticmethod
    def _video_frame_count(video_path):
        """返回视频总帧数；失败返回 None。"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return total if total > 0 else None
        except Exception:
            return None

    @staticmethod
    def _set(wf, node_id, key, value):
        if not node_id or node_id not in wf:
            return False
        inp = wf[node_id].setdefault("inputs", {})
        if key not in inp:
            return False
        inp[key] = value
        return True
