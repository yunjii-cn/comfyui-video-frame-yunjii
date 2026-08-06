"""
Tier 2 暖启动适配器（AnimatePlus SCAIL-2 路线）。

设计目标：把 SCAIL-2 生成从 WanVideoWrapper 家族重包进 WanAnimatePlus 家族
（复用用户现成的「分段队列」丝滑工作流），用其原生时序建模 + 多段重叠融合得到
「连续 + 画质」兼得的真·一镜到底；并对 seg>0 把上段真实成片末 5 帧经
`VHS_LoadVideo` 截帧后注入 `WanAnimatePlus SCAIL_2 Embeds.prefix_frames`，
做跨段暖启动（硬冻结到本段 latent 开头，6 个 prefix latent 预置）。

复用：
- SCAILAdapter 的全部静态手术（Set/Get 重连、Reroute 解析、未注册节点清理、
  UI→API 转换、模型名模糊匹配）——与具体家族无关，通用。本适配器直接继承
  SCAILAdapter，从而自然复用这些实例方法，只覆盖 discover / modify / prepare。

非破坏性：若模板不是 WanAnimatePlus SCAIL_2 工作流（缺关键节点），discover 失败，
runner 会优雅回退到标准 SCAIL 路线；prefix 注入任何异常都被 try/except 吞掉，
最坏情况只是「无 prefix 的 WanAnimatePlus 多段」（仍比纯 WanVideoWrapper 连续）。

注意：本适配器端到端跑通依赖本机 WanAnimatePlus 节点集 + 权重 + 显存，需用户在
GPU 上验证。若报错，回退「多段无缝」或「单遍连贯(方案C)」即可，现有 WanVideoWrapper
SCAIL 路线不受影响。
"""

import os
import json
import cv2

from .scail import SCAILAdapter, DEFAULT_NEGATIVE
from ..debug_log import info, warn, error as log_error

# WanAnimatePlus SCAIL-2 Embeds 节点的两种 class_type（本机运行版本 vs 磁盘新版）
AP_SCAIL_EMBEDS_VARIANTS = ("WanAnimatePlus SCAIL_2 Embeds", "WanAnimatePlusSCAIL2Embeds")
AP_SAMPLER_FROM_SETTINGS = "WanAnimatePlus SamplerFromSettings"
AP_SAMPLER_SETTINGS = "WanAnimatePlus SamplerSettings"
AP_SAMPLER = "WanAnimatePlus Sampler"
AP_MODEL_LOADER = "WanAnimatePlus ModelLoader"
AP_CONTEXT_OPTIONS = "WanAnimatePlus ContextOptions"
AP_TEXT_ENCODE_VARIANTS = ("WanAnimatePlus TextEncodeCached", "WanVideoTextEncodeCached")
AP_VHS_LOADVIDEO = "VHS_LoadVideo"
AP_LOADIMAGE = "LoadImage"
AP_VIDEO_COMBINE = "VHS_VideoCombine"
AP_VAE_LOADER_VARIANTS = ("WanAnimatePlus VAELoader", "WanVideo VAELoader")
AP_CLIP_VISION_VARIANTS = ("WanAnimatePlus ClipVisionEncode V2", "WanVideoClipVisionEncode")

# prefix_frames 最多注入帧数（WanAnimatePlus 限制 ≤5，最多 17 帧 prefix）
PREFIX_MAX_FRAMES = 5


class AnimatePlusNodeMap:
    """Tier 2 节点映射（不继承 NodeMap，字段按本家族定制）。"""

    def __init__(self):
        self.animate_embeds = ""
        self.sampler = ""
        self.sampler_settings = ""
        self.model_loader = ""
        self.context_options = ""
        self.text_encode = ""
        self.vae_loader = ""
        self.clip_vision = ""
        self.driving_video = ""
        self.ref_image = ""
        self.video_combine = ""
        # 潜空间拼接基建复用：SCAILAdapter._inject_save_latent / _extract_decode_template
        # 依赖 node_map.decode 与 node_map.combine。AnimatePlus 家族原 NodeMap 缺这俩字段，
        # 导致 WanAnimatePlus 路线的 latent 落盘/解码模板抽取静默失败(AttributeError 被吞)，
        # 潜空间拼接永远回退像素拼接。此处补齐，使『真骨架多段 + 潜空间拼接』在 WanAnimatePlus
        # 模板上也能生效(与标准 SCAIL 一致)。
        self.decode = ""
        self.combine = ""

    def to_dict(self):
        return {
            "animate_embeds": self.animate_embeds,
            "sampler": self.sampler,
            "sampler_settings": self.sampler_settings,
            "model_loader": self.model_loader,
            "context_options": self.context_options,
            "text_encode": self.text_encode,
            "vae_loader": self.vae_loader,
            "clip_vision": self.clip_vision,
            "driving_video": self.driving_video,
            "ref_image": self.ref_image,
            "video_combine": self.video_combine,
        }

    def is_valid(self):
        return bool(self.animate_embeds and self.sampler and self.video_combine
                    and self.driving_video and self.ref_image)


class AnimatePlusSCAILAdapter(SCAILAdapter):
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
        for nid, ndata in workflow.items():
            if not isinstance(ndata, dict):
                continue
            ct = ndata.get("class_type") or ndata.get("type", "")
            if ct in AP_SCAIL_EMBEDS_VARIANTS and not nm.animate_embeds:
                nm.animate_embeds = nid
            elif ct in (AP_SAMPLER_FROM_SETTINGS, AP_SAMPLER) and not nm.sampler:
                nm.sampler = nid
            elif ct == AP_SAMPLER_SETTINGS and not nm.sampler_settings:
                nm.sampler_settings = nid
            elif ct == AP_MODEL_LOADER and not nm.model_loader:
                nm.model_loader = nid
            elif ct == AP_CONTEXT_OPTIONS and not nm.context_options:
                nm.context_options = nid
            elif ct in AP_TEXT_ENCODE_VARIANTS and not nm.text_encode:
                nm.text_encode = nid
            elif ct in AP_VAE_LOADER_VARIANTS and not nm.vae_loader:
                nm.vae_loader = nid
            elif ct in AP_CLIP_VISION_VARIANTS and not nm.clip_vision:
                nm.clip_vision = nid
            elif ct == AP_VIDEO_COMBINE:
                prefix, save_out = self._vhs_meta(ndata)
                vhs_candidates.append((nid, prefix, save_out))
            elif ct in ("WanVideoDecode", "WanAnimatePlus Decode") and not nm.decode:
                # 解码节点（端口 samples/vae 与标准 WanVideoDecode 一致），供潜空间拼接落盘/解码
                nm.decode = nid

        # 驱动视频：沿 embeds.pose_images 向上回溯，找第一个 VHS_LoadVideo
        if nm.animate_embeds and nm.animate_embeds in workflow:
            nm.driving_video = self._trace_to_class(
                workflow, nm.animate_embeds, "pose_images", AP_VHS_LOADVIDEO)
        # 参考图：沿 embeds.ref_image 向上回溯，找第一个 LoadImage
        if nm.animate_embeds and nm.animate_embeds in workflow:
            nm.ref_image = self._trace_to_class(
                workflow, nm.animate_embeds, "ref_image", AP_LOADIMAGE)

        nm.video_combine = self._select_primary_vhs(vhs_candidates)
        # 供 SCAILAdapter 潜空间落盘/解码复用：combine 取主成片 VHS，decode 取解码节点
        nm.combine = nm.video_combine
        if not nm.is_valid():
            miss = [n for n, v in [
                ("WanAnimatePlus SCAIL_2 Embeds", nm.animate_embeds),
                ("WanAnimatePlus Sampler", nm.sampler),
                ("VHS_VideoCombine", nm.video_combine),
                ("VHS_LoadVideo(驱动)", nm.driving_video),
                ("LoadImage(参考)", nm.ref_image),
            ] if not v]
            warn("AnimatePlusAdapter", "工作流缺少必要 WanAnimatePlus 节点: %s", ", ".join(miss))
        return nm

    def _trace_to_class(self, workflow, start_node, start_input, target_class, max_depth=12):
        """从 start_node 的 start_input 链接出发，沿输入链向上回溯，
        返回第一个 class_type==target_class 的祖先节点 id；找不到返回 ''。"""
        node = workflow.get(start_node)
        if not isinstance(node, dict):
            return ""
        link = node.get("inputs", {}).get(start_input)
        if not (isinstance(link, list) and len(link) >= 1):
            return ""
        seen, stack = set(), [str(link[0])]
        while stack and len(seen) < max_depth + 1:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            nd = workflow.get(nid)
            if not isinstance(nd, dict):
                continue
            if nd.get("class_type") == target_class:
                return nid
            for v in nd.get("inputs", {}).values():
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
            # 已是 API 格式
            api = self._prune_api_missing(dict(workflow_raw))
            self._sanitize_numeric_inputs(api)
            self._fix_model_names(api)
            return api
        # UI 完整格式：手术 + 转换（与 SCAILAdapter 同套通用清理，但不跑
        # WanVideoWrapper 专属的 _fix_pose_images / 姿态分支丢弃）
        full = json.loads(json.dumps(workflow_raw))
        full = self._rewire_setget(full)
        full = self._resolve_reroutes(full)
        full = self._delete_unregistered(full)
        full = self._delete_dangling_vhs(full)
        full = self._keep_main_chain(full)
        full = self._drop_bypassed(full)
        api = self._convert_full_to_api(full, self._node_class_mappings())
        self._sanitize_numeric_inputs(api)
        self._fix_model_names(api)
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

        # 0) 主输出节点强制落盘（参考工作流里 VHS_VideoCombine.save_output 可能为 False，
        #    不强制会导致 ComfyUI 不写文件、runner 抓不到成片路径）
        if node_map.video_combine and node_map.video_combine in wf:
            vc = wf[node_map.video_combine].setdefault("inputs", {})
            vc["save_output"] = True
            # 清理 VHS 输出前缀：模板常带 %date:yyyy-MM-dd% 等 Windows 非法路径 token，
            # ComfyUI 不展开，直接传给 os.makedirs 会因 ':' 非法而崩溃(WinError 267)。
            # 统一改写成安全、分段可定位的命名（与主线 yunjii_v2v 同源风格）。
            vc["filename_prefix"] = "yunjii_tier2/seg%d" % seg.index

        # 1) 参考图（每段注入，保证身份一致）
        if char_ref and node_map.ref_image and node_map.ref_image in wf:
            img_name = self._copy_to_input(char_ref)
            if img_name:
                wf[node_map.ref_image].setdefault("inputs", {})["image"] = img_name
                info("AnimatePlusAdapter", "参考图注入: node=%s, image=%s", node_map.ref_image, img_name)

        # 2) WanAnimatePlus SCAIL_2 Embeds：帧数（4k+1 对齐）+ frame_window_size
        if node_map.animate_embeds and node_map.animate_embeds in wf:
            ae = wf[node_map.animate_embeds]
            ae.setdefault("inputs", {})
            n = seg.target_frames
            aligned = max(9, ((n - 1) // 4) * 4 + 1)
            ae["inputs"]["num_frames"] = aligned
            if "frame_window_size" in ae["inputs"]:
                ae["inputs"]["frame_window_size"] = aligned
            info("AnimatePlusAdapter", "Embeds: num_frames=%d(aligned %d), 段索引=%d",
                 n, aligned, seg.index)

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
            info("AnimatePlusAdapter", "驱动视频: %s, 偏移=%d, 段长=%d",
                 fname, seg.start_frame, seg.target_frames)
        elif node_map.driving_video and not (self.driving_video_path
                                             and os.path.isfile(self.driving_video_path)):
            warn("AnimatePlusAdapter", "未设置驱动视频路径，WanAnimatePlus SCAIL-2 无法生成动作")

        # 4) 提示词
        if node_map.text_encode and node_map.text_encode in wf:
            neg = (seg.params.get("negative") if isinstance(seg.params, dict) else None) \
                or DEFAULT_NEGATIVE
            self._set(wf, node_map.text_encode, "positive_prompt", seg.prompt or "")
            self._set(wf, node_map.text_encode, "negative_prompt", neg)

        # 5) 上下文窗口随段长收缩（防御性：不超过 num_frames）
        if node_map.context_options and node_map.context_options in wf:
            co = wf[node_map.context_options].setdefault("inputs", {})
            if "context_frames" in co:
                cur = co["context_frames"]
                try:
                    cur = int(cur)
                except (TypeError, ValueError):
                    cur = 81
                co["context_frames"] = min(cur, seg.target_frames) if seg.target_frames < cur else cur

        # 6) 模型精度（仅 fp16 时调整 base_precision；fp8 保持工作流默认）
        if self.model_precision == "fp16" and node_map.model_loader and node_map.model_loader in wf:
            ml = wf[node_map.model_loader].setdefault("inputs", {})
            if "base_precision" in ml:
                ml["base_precision"] = "fp16_fast"
            if "quantization" in ml:
                ml["quantization"] = "disabled"

        # 7) prefix_frames 暖启动（seg>0）：上段成片末帧 → 硬冻结到本段 latent 开头
        if seg.index > 0 and prev_video_path and os.path.isfile(prev_video_path):
            self._inject_prefix(wf, node_map, prev_video_path)

        # 蒸馏 LoRA 路线：钉模型/LoRA 到本机真实文件（覆盖 WanAnimatePlus 系列节点
        # 的 model / lora_0..N 字段——用户直跑工作流用 WanAnimatePlus LoraSelectMulti
        # 挂蒸馏 LoRA，旧路径 Wan-Lighting\... 在本机不存在，必须钉到 wan\..._rank256）。
        self._pin_distill_lora_and_model(wf)
        # 防御：SCAIL 基座模型低步数 → 模糊；若挂载蒸馏 LoRA 则强制 4 步(见 _enforce 内)
        wf = self._enforce_min_sampling_steps(wf)
        return wf

    # ------------------------------------------------------------------
    # prefix 注入（best-effort，任何异常都吞掉，保证工作流仍可跑）
    # ------------------------------------------------------------------
    def _inject_prefix(self, wf, node_map, prev_video_path):
        try:
            if not node_map.animate_embeds or node_map.animate_embeds not in wf:
                return
            ae = wf[node_map.animate_embeds]
            # 该节点类（WanAnimatePlus SCAIL_2 Embeds）原生支持 prefix_frames。
            # 未连线的可选输入在 UI→API 转换后不会出现在 inputs 里，此时应直接注入；
            # 仅当模板已硬占用 prefix_frames（非 None）时才跳过。
            ct = ae.get("class_type", "")
            if ct not in AP_SCAIL_EMBEDS_VARIANTS:
                return  # 未知节点，不冒险注入
            if ae.get("inputs", {}).get("prefix_frames") is not None:
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

            # 直接让 VHS 加载最后 n 帧（skip_first_frames = max(0,total-n), cap=n），
            # 输出 IMAGE 直连 prefix_frames，避免 ImageFromBatch 负索引歧义。
            vhs_id = "yunjii_prev_vhs_%d" % (abs(hash(prev_video_path)) % 100000)
            wf[vhs_id] = {
                "class_type": AP_VHS_LOADVIDEO,
                "inputs": {
                    "video": fname,
                    "force_rate": 0,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": n,
                    "skip_first_frames": max(0, (total or n) - n),
                    "select_every_nth": 1,
                    "format": "Wan",
                },
            }
            ae["inputs"]["prefix_frames"] = [vhs_id, 0]
            info("AnimatePlusAdapter", "prefix_frames 注入: 上段成片%s → 末%d帧暖启动", prev_video_path, n)
        except Exception as e:
            warn("AnimatePlusAdapter", "prefix_frames 注入异常(已跳过，不影响主生成): %s", str(e)[:200])

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
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
