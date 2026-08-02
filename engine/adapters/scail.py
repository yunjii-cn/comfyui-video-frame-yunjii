"""
SCAIL-2 生成后端适配器（重写版：驱动官方 embeds 子流程 / 方案 B）。

背景与兼容性（2026-07-28 实时 M0 审计结论）：
- SCAIL-2 不是单节点，而是插在 WanVideo 管线里的子流程。官方 example 用的是
  "embeds 方案"：驱动视频 → NLF/ViTPose 姿态 → WanVideoAddSCAILPoseEmbeds，
  参考图 → WanVideoAddSCAILReferenceEmbeds（含 WanVideoClipVisionEncode 出的 clip_embeds），
  二者注入 WanVideoEmptyEmbeds 后再进 WanVideoSamplerv2。本适配器就驱动这条官方子流程。
- 早期占位版（SCAIL_FIELD_MAP / WanSCAILToVideo 假设）字段全错，已废弃。

本适配器复用 DirectAdapter 的内联执行核心（init_executor / execute_inline /
cleanup_executor / _copy_to_input / _extract_video_from_history 等），只重写：
  - discover_nodes               : 识别官方 SCAIL 子流程各节点
  - modify_workflow_for_segment  : 把 planner 的 SegmentInfo 映射到工作流输入

注入点（键名均来自运行 ComfyUI 的 object_info 实测）：
  - WanVideoModelLoader.model           : 自动探测 diffusion_models 下 *SCAIL*fp8* 权重
  - VHS_LoadVideo.video/skip_first_frames/frame_load_cap : 驱动视频 + 分段偏移 + 段长
  - LoadImage.image (参考图上游)        : 角色参考图（每段都注入，保证身份一致）
  - WanVideoTextEncodeCached.positive_prompt/negative_prompt : 提示词
  - WanVideoEmptyEmbeds.width/height/num_frames           : 分辨率与帧数
  - WanVideoSamplerv2.seed             : 每段可复现
  其余（VAE 加载、CLIP Vision、SCAIL 强度、block_swap、LoRA、scheduler）沿用官方工作流默认。

长视频处理：每段独立用同一参考图 + 驱动视频的对应片段生成，runner 负责段间拼接
（ffmpeg/cv2），与骨骼路线一致；不依赖 WanSCAILToVideo 的 previous_frames 串联。
"""

import os
import json
import glob

from .direct import DirectAdapter
from ..debug_log import info, warn, error as log_error

# 官方子流程用到的 class_type
SCAIL_REF_EMBEDS = "WanVideoAddSCAILReferenceEmbeds"
SCAIL_POSE_EMBEDS = "WanVideoAddSCAILPoseEmbeds"
SCAIL_MODEL_LOADER = "WanVideoModelLoader"
SCAIL_EMPTY_EMBEDS = "WanVideoEmptyEmbeds"
SCAIL_TEXT_ENCODE = "WanVideoTextEncodeCached"
SCAIL_DRIVING_VIDEO = "VHS_LoadVideo"
SCAIL_REF_IMAGE = "LoadImage"
SCAIL_SAMPLER = "WanVideoSamplerv2"
SCAIL_DECODE = "WanVideoDecode"
SCAIL_COMBINE = "VHS_VideoCombine"

# 核心 SCAIL 链路节点：预处理时即使有必填输入悬空也绝不自动删除，
# 只把悬空输入置空，交给下游回退（如 pose_images 接驱动帧）。
PROTECTED_CLASS_TYPES = {
    SCAIL_REF_EMBEDS, SCAIL_POSE_EMBEDS, SCAIL_MODEL_LOADER,
    SCAIL_EMPTY_EMBEDS, SCAIL_TEXT_ENCODE, SCAIL_DRIVING_VIDEO,
    SCAIL_REF_IMAGE, SCAIL_SAMPLER, SCAIL_DECODE, SCAIL_COMBINE,
}

# SCAIL 14B fp8 权重在 diffusion_models 下的探测通配（兼容 Comfy-Org 与 KJ 两种命名）
SCAIL_MODEL_GLOB = "*SCAIL*fp8*"
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景"
)


class SCAILNodeMap:
    """官方 SCAIL 子流程的节点映射。"""

    def __init__(self):
        self.model_loader = ""
        self.text_encode = ""
        self.scail_ref_embeds = ""
        self.scail_pose_embeds = ""
        self.empty_embeds = ""
        self.driving_video = ""
        self.reference_image = ""   # 参考图上游 LoadImage 节点 id
        self.sampler = ""
        self.decode = ""
        self.combine = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v}

    def is_valid(self):
        need = [
            self.model_loader, self.text_encode, self.scail_ref_embeds,
            self.scail_pose_embeds, self.empty_embeds, self.driving_video,
            self.decode, self.combine,
        ]
        return all(need)


class SCAILAdapter(DirectAdapter):
    """SCAIL-2（embeds 方案 B）生成后端。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 由 runner 在 SCAIL 路线注入：动作源视频（驱动视频）路径
        self.driving_video_path = ""
        # 模型精度：fp8(默认,省显存) / fp16(更精细,吃显存)。通过选不同权重文件实现。
        self.model_precision = "fp8"

    def _detect_scail_model(self, precision="fp8"):
        """在 diffusion_models 目录下探测 SCAIL 权重，返回相对路径。

        precision="fp8"（默认）：选 *SCAIL*fp8* 权重（省显存，RTX3090 稳定）。
        precision="fp16"：优先选非 fp8 的 SCAIL 权重（fp16/bf16，更精细）；
                          若本机无此类权重则回退 fp8 并告警（不报错，保证可跑）。
        """
        try:
            import folder_paths
            dirs = folder_paths.get_folder_paths("diffusion_models") or []
        except Exception:
            dirs = []
        if not dirs:
            dirs = [os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "models", "diffusion_models")]

        if precision == "fp16":
            for d in dirs:
                # 排除显式 fp8 文件，优先取其他 SCAIL 权重（fp16/bf16）
                all_scail = glob.glob(os.path.join(d, "*SCAIL*"), recursive=True)
                if not all_scail:
                    all_scail = glob.glob(os.path.join(d, "**", "*SCAIL*"), recursive=True)
                non_fp8 = [m for m in all_scail
                           if "fp8" not in os.path.basename(m).lower()]
                if non_fp8:
                    full = non_fp8[0]
                    rel = os.path.relpath(full, d)
                    info("SCAILAdapter", "SCAIL 权重(fp16): %s", rel)
                    return rel.replace(os.sep, "/")
            warn("SCAILAdapter", "未探测到非-fp8 的 SCAIL 权重，回退 fp8")

        for d in dirs:
            matches = glob.glob(os.path.join(d, SCAIL_MODEL_GLOB), recursive=True)
            if not matches:
                matches = glob.glob(os.path.join(d, "**", SCAIL_MODEL_GLOB), recursive=True)
            if matches:
                full = matches[0]
                rel = os.path.relpath(full, d)
                return rel.replace(os.sep, "/")
        return None

    def _find_upstream_loadimage(self, workflow, start_node_id, max_depth=10):
        """从某节点沿 link([src_node, slot]) 向上回溯，返回第一个 LoadImage 的 id。

        用于定位参考图源：官方 SCAIL 工作流里 ref_embeds.ref_image 往往经过
        Reroute / ImageResize 等中间节点才连到真正的 LoadImage，需多跳回溯。
        """
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
            if node.get("class_type") == SCAIL_REF_IMAGE:
                return nid
            for v in node.get("inputs", {}).values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    stack.append(v[0])
        return ""

    def discover_nodes(self, workflow):
        nm = SCAILNodeMap()
        items = DirectAdapter._iter_nodes(workflow)  # 兼容 nodes 为 list 或 dict
        vhs_candidates = []
        for nid, ndata in items:
            if not isinstance(ndata, dict):
                continue
            ct = ndata.get("class_type") or ndata.get("type", "")
            if ct == SCAIL_MODEL_LOADER and not nm.model_loader:
                nm.model_loader = nid
            elif ct == SCAIL_TEXT_ENCODE and not nm.text_encode:
                nm.text_encode = nid
            elif ct == SCAIL_REF_EMBEDS and not nm.scail_ref_embeds:
                nm.scail_ref_embeds = nid
            elif ct == SCAIL_POSE_EMBEDS and not nm.scail_pose_embeds:
                nm.scail_pose_embeds = nid
            elif ct == SCAIL_EMPTY_EMBEDS and not nm.empty_embeds:
                nm.empty_embeds = nid
            elif ct == SCAIL_DRIVING_VIDEO and not nm.driving_video:
                nm.driving_video = nid
            elif ct == SCAIL_SAMPLER and not nm.sampler:
                nm.sampler = nid
            elif ct == SCAIL_DECODE and not nm.decode:
                nm.decode = nid
            elif ct == SCAIL_COMBINE:
                # 收集所有 VHS，最后统一挑选真实成片（剔除姿态/骨架预览节点）
                prefix, save_out = DirectAdapter._vhs_meta(ndata)
                vhs_candidates.append((nid, prefix, save_out))
            elif ct == SCAIL_REF_IMAGE and not nm.reference_image:
                title = (ndata.get("title") or ndata.get("_meta", {}).get("title") or "")
                if "reference" in title.lower():
                    nm.reference_image = nid

        # 主输出：剔除 onetotall_pose 等姿态/骨架预览节点，优先 save_output=True 的真实成片
        nm.combine = DirectAdapter._select_primary_vhs(vhs_candidates)

        # 参考图常经 Reroute / ImageResize 等节点间接连到 ref_image
        # (官方工作流: LoadImage -> ImageResizeKJv2 -> Reroute -> ref_embeds.ref_image)，
        # 故从 scail_ref_embeds.ref_image 沿 link 上游回溯，找真正的 LoadImage(用户参考图源)
        if not nm.reference_image and nm.scail_ref_embeds and nm.scail_ref_embeds in workflow:
            ref_link = workflow[nm.scail_ref_embeds].get("inputs", {}).get("ref_image")
            if isinstance(ref_link, list) and len(ref_link) >= 1:
                li = self._find_upstream_loadimage(workflow, ref_link[0])
                if li:
                    nm.reference_image = li

        if not nm.is_valid():
            miss = [name for name, val in [
                ("WanVideoModelLoader", nm.model_loader),
                ("WanVideoTextEncodeCached", nm.text_encode),
                ("WanVideoAddSCAILReferenceEmbeds", nm.scail_ref_embeds),
                ("WanVideoAddSCAILPoseEmbeds", nm.scail_pose_embeds),
                ("WanVideoEmptyEmbeds", nm.empty_embeds),
                ("VHS_LoadVideo", nm.driving_video),
                ("WanVideoDecode", nm.decode),
                ("VHS_VideoCombine", nm.combine),
            ] if not val]
            warn("SCAILAdapter", "工作流缺少必要 SCAIL-2 节点: %s", ", ".join(miss))
        return nm

    def _set(self, wf, node_id, key, value):
        if not node_id or node_id not in wf:
            return False
        inp = wf[node_id].setdefault("inputs", {})
        if key not in inp:
            warn("SCAILAdapter", "节点 %s 无输入键 '%s'，跳过", node_id, key)
            return False
        inp[key] = value
        return True

    def modify_workflow_for_segment(self, workflow, node_map, seg, ref_image_path, pose_dir="", run_id="", user_ref_path=""):
        """
        把一个 SegmentInfo 映射到官方 SCAIL 子流程输入。
        workflow 为 API 格式（链接以 [node, slot] 表示，未链接的 widget 为原始值）。
        参考图用 user_ref_path（角色身份，每段一致）；驱动视频用 self.driving_video_path。
        """
        wf = json.loads(json.dumps(workflow))
        # SCAIL 身份一致性：始终用角色参考图，不随段切换到前段末帧
        char_ref = user_ref_path or ref_image_path

        # 1) 模型权重：自动探测 SCAIL 权重（fp16 优先非-fp8 文件，否则回退 fp8）
        scail_model = self._detect_scail_model(self.model_precision)
        if scail_model:
            self._set(wf, node_map.model_loader, "model", scail_model)
            info("SCAILAdapter", "SCAIL 权重: %s", scail_model)
        else:
            warn("SCAILAdapter", "未探测到 SCAIL fp8 权重(*SCAIL*fp8*)，请确认已下载到 diffusion_models/")

        # 2) 角色参考图：注入上游 LoadImage（每段都注入，保证身份一致）
        if char_ref and node_map.reference_image:
            img_name = self._copy_to_input(char_ref)
            if img_name and self._set(wf, node_map.reference_image, "image", img_name):
                info("SCAILAdapter", "参考图注入: node=%s, image=%s", node_map.reference_image, img_name)

        # 3) 驱动视频（动作源）+ 分段偏移 + 段长
        if node_map.driving_video and self.driving_video_path and os.path.isfile(self.driving_video_path):
            fname = self._copy_to_input(self.driving_video_path)
            di = wf[node_map.driving_video].setdefault("inputs", {})
            if fname and "video" in di:
                di["video"] = fname
            di["skip_first_frames"] = max(0, seg.start_frame)
            di["frame_load_cap"] = seg.target_frames
            if "select_every_nth" in di:
                di["select_every_nth"] = 1
            info("SCAILAdapter", "驱动视频: %s, 偏移=%d, 段长=%d", fname, seg.start_frame, seg.target_frames)
        elif node_map.driving_video:
            warn("SCAILAdapter", "未设置驱动视频路径(driving_video_path)，SCAIL-2 无法生成动作", )

        # 4) 提示词（真实键名 positive_prompt / negative_prompt）
        if node_map.text_encode:
            self._set(wf, node_map.text_encode, "positive_prompt", seg.prompt or "")
            neg = seg.params.get("negative") if isinstance(seg.params, dict) else None
            self._set(wf, node_map.text_encode, "negative_prompt", neg or DEFAULT_NEGATIVE)

        # 5) 分辨率 + 帧数
        w = (seg.params.get("width", 480) if isinstance(seg.params, dict) else 480)
        h = (seg.params.get("height", 832) if isinstance(seg.params, dict) else 832)
        n = seg.target_frames
        self._set(wf, node_map.empty_embeds, "width", w)
        self._set(wf, node_map.empty_embeds, "height", h)
        self._set(wf, node_map.empty_embeds, "num_frames", n)

        # 5.5) 尺寸常量同步：官方工作流用 INTConstant(203=width,204=height)
        # 决定参考图预处理尺寸，必须与注入的 width/height 一致，否则参考图 latent
        # 与 empty_embeds 尺寸不匹配（实测报 Expected 112 but got 104）。
        for cid, role in (("203", "w"), ("204", "h")):
            node = wf.get(cid)
            if node and node.get("class_type") == "INTConstant":
                target = w if role == "w" else h
                if isinstance(target, int) and node["inputs"].get("value") != target:
                    node["inputs"]["value"] = target
                    info("SCAILAdapter", "尺寸常量 INTConstant %s -> %d", cid, target)

        # 6) 采样种子：每段可复现
        if node_map.sampler:
            seed = (1234567 + seg.index * 101) & 0xFFFFFFFF
            self._set(wf, node_map.sampler, "seed", seed)

        info("SCAILAdapter", "核心参数: width=%d, height=%d, num_frames=%d, seed=%d",
             w, h, n, (1234567 + seg.index * 101) & 0xFFFFFFFF)

        # 蒸馏 LoRA 路线：固定 4 步快速（步数蒸馏 LoRA 为 4 步设计，高步数反而崩坏）。
        # 同时把模型/LoRA 显式钉到本机真实存在的文件，避免模板错位文件(不存在的
        # preview 模型 / rank64 蒸馏 LoRA 名)作怪。
        if SCAILAdapter._workflow_has_distill_lora(wf):
            # 蒸馏 LoRA 路线：凡是带 steps 输入的采样/调度节点，一律钉到 4 步(快速)。
            # 步数蒸馏 LoRA 为 4 步设计，高步数会因 schedule 错配产生结构性崩坏伪影。
            # 覆盖两类挂载方式：调度器(WanVideoSchedulerv2) 与 采样器设置
            # (WanAnimatePlus SamplerSettings / WanVideoSamplerv2)，两条路线都修。
            for nid, nd in wf.items():
                if not isinstance(nd, dict):
                    continue
                ct = nd.get("class_type") or ""
                if not any(k in ct for k in ("Schedulerv2", "Scheduler",
                                             "SamplerSettings", "Samplerv2", "Sampler")):
                    continue
                inp = nd.get("inputs")
                if isinstance(inp, dict) and "steps" in inp:
                    if int(inp.get("steps", 0)) != 4:
                        inp["steps"] = 4
                        info("SCAILAdapter", "蒸馏 LoRA 路线: %s(%s) 步数→4(快速)", nid, ct)
            self._pin_distill_lora_and_model(wf)

        # 防御：SCAIL 基座模型低步数 → 模糊（详见 _enforce_min_sampling_steps）
        wf = self._enforce_min_sampling_steps(wf)
        return wf

    @staticmethod
    def _pin_distill_lora_and_model(wf):
        """蒸馏 LoRA 路线：把模型与蒸馏 LoRA 显式钉到本机真实存在的文件。

        背景：官方 SCAIL 模板节点默认指向 Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ
        (模型) 与 lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16 (LoRA)，二者在本机
        均不存在；_detect_scail_model 已把模型替成 wan2.1_14B_SCAIL_2_fp8_scaled，但
        LoRA 仅靠 _fix_model_names 模糊匹配，可能因同名权重多而选错。这里显式锁定：
        模型 = wan2.1_14B_SCAIL_2_fp8_scaled，蒸馏 LoRA = lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16。
        """
        import os as _os
        dm_dirs = []
        try:
            import folder_paths
            dm_dirs = folder_paths.get_folder_paths("diffusion_models") or []
        except Exception:
            pass
        if not dm_dirs:
            dm_dirs = [r"F:\ComfyUI_heihe\ComfyUI\models\diffusion_models"]
        lora_dirs = []
        try:
            lora_dirs = folder_paths.get_folder_paths("loras") or []
        except Exception:
            pass
        if not lora_dirs:
            lora_dirs = [r"F:\ComfyUI_heihe\ComfyUI\models\loras"]

        target_model = "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"
        target_lora = _os.path.join("wan", "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors")

        def _exists(name, dirs):
            for d in dirs:
                if _os.path.isfile(_os.path.join(d, name)):
                    return True
            return False

        if not _exists(target_model, dm_dirs):
            warn("SCAILAdapter", "钉模型失败: %s 不在 diffusion_models/", target_model)
            target_model = None
        if not _exists(target_lora, lora_dirs):
            warn("SCAILAdapter", "钉 LoRA 失败: %s 不在 loras/", target_lora)
            target_lora = None

        for nid, nd in wf.items():
            if not isinstance(nd, dict):
                continue
            ct = nd.get("class_type") or ""
            inp = nd.get("inputs")
            if not isinstance(inp, dict):
                continue
            # 模型加载器：钉 model 字段（覆盖 WanVideo / WanAnimatePlus 两种 ModelLoader）
            if "ModelLoader" in ct and target_model and "model" in inp:
                if inp["model"] != target_model:
                    inp["model"] = target_model
                    info("SCAILAdapter", "钉模型: %s → %s", nid, target_model)
            # LoRA 选择节点：任何含 'lora' 的输入字段，凡值是 step-distill 即钉到真实蒸馏 LoRA。
            # 覆盖 WanVideoLoraSelect(lora) / WanVideoLoraSelectByName(lora_name) /
            # WanAnimatePlus LoraSelectMulti(lora_0..lora_N) 等任意字段命名。
            if target_lora and ("LoraSelect" in ct or "SetLoRA" in ct):
                for fk, fv in list(inp.items()):
                    if not (isinstance(fk, str) and "lora" in fk.lower()):
                        continue
                    if isinstance(fv, str) and "distill" in fv.lower() and fv != target_lora:
                        inp[fk] = target_lora
                        info("SCAILAdapter", "钉蒸馏 LoRA: %s.%s → %s", nid, fk, target_lora)

    # ==================================================================
    # 工作流预处理：把官方 SCAIL UI 工作流整理成干净、可提交的 API 工作流。
    # 这些修复原只在验证脚本里，现并入生产适配器，使 SCAIL-2 路线在本机
    # 节点集 / 模型路径 / 尺寸常量 / 姿态插件存在差异时也能跑通。
    # ==================================================================

    @staticmethod
    def _object_info():
        """返回运行中 ComfyUI 的 /object_info（权威：含全部自定义节点）。
        缓存一次，避免每次调用都打 HTTP。独立运行或与本机服务通信时都可用。"""
        cache = getattr(SCAILAdapter, "_OI_CACHE", None)
        if cache is not None:
            return cache
        try:
            import urllib.request, json as _json
            # 强制不走代理（localhost 服务），否则会被 HTTP_PROXY 路由到 127.0.0.1:7890 而失败
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with op.open("http://127.0.0.1:8188/object_info", timeout=20) as r:
                oi = _json.load(r)
            if oi:
                SCAILAdapter._OI_CACHE = oi
                return oi
        except Exception:
            pass
        # 兜底：本进程已注册节点（生产环境 runner 内联执行时等价于 object_info）。
        # 仅返回 class_type 集合占位；widget/required 细节在 object_info 不可用时退化为空（安全）。
        try:
            import nodes
            if getattr(nodes, "NODE_CLASS_MAPPINGS", None):
                return {k: {} for k in nodes.NODE_CLASS_MAPPINGS.keys()}
        except Exception:
            pass
        return {}

    @staticmethod
    def _node_class_mappings():
        """返回本机已注册 class_type 集合（dict-like）。
        优先用运行中的 ComfyUI /object_info（含全部自定义节点，最权威）；
        独立 import nodes 往往只含核心节点，故仅作兜底。"""
        oi = SCAILAdapter._object_info()
        if oi:
            return {k: {} for k in oi.keys()}
        try:
            import nodes
            if getattr(nodes, "NODE_CLASS_MAPPINGS", None):
                return nodes.NODE_CLASS_MAPPINGS
        except Exception:
            pass
        return {}

    @staticmethod
    def _widget_names(class_type):
        """返回某 class_type 的 widget 输入名列表（兼容 dict/list 两种 widgets_values）。"""
        oi = SCAILAdapter._object_info()
        info = oi.get(class_type)
        if not info:
            return []
        names = []
        for cat in ("required", "optional"):
            for name, cfg in info.get("input", {}).get(cat, {}).items():
                if not (isinstance(cfg, list) and cfg):
                    continue
                typ = cfg[0]
                if (isinstance(typ, str) and typ in
                     ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO", "ENUM")) \
                        or isinstance(typ, list):
                    names.append(name)
        return names

    @staticmethod
    def _rewire_setget(full):
        """KJNodes Set/Get 节点在本机版本未注册 -> POST 400。
        把每个 Get 节点的消费方直接重连到对应 Set 节点的源值，并删除 Set/Get。"""
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        link_src = {}
        for l in links:
            if len(l) >= 5:
                link_src[l[0]] = (str(l[1]), l[2])
        sets, gets, sg_ids = {}, {}, set()
        for n in nodes:
            t = n.get("type")
            if t in ("SetNode", "GetNode"):
                sg_ids.add(str(n["id"]))
                wv = n.get("widgets_values") or []
                name = wv[0] if wv else None
                if t == "SetNode":
                    ins = n.get("inputs", [])
                    lnk = ins[0].get("link") if ins else None
                    if lnk is not None and lnk in link_src:
                        sets[name] = link_src[lnk]
                else:
                    for o in n.get("outputs", []):
                        for lid in (o.get("links") or []):
                            gets.setdefault(name, []).append(lid)
        link_by_id = {l[0]: l for l in links if len(l) >= 5}
        for name, glinks in gets.items():
            src = sets.get(name)
            if not src:
                continue
            for gl in glinks:
                if gl in link_by_id:
                    link_by_id[gl][1] = int(src[0])
                    link_by_id[gl][2] = src[1]
        full["nodes"] = [n for n in nodes if str(n["id"]) not in sg_ids]
        info("SCAILAdapter", "预处理：移除 %d 个 Set/Get 节点并重连 %d 条 Get 边",
              len(sg_ids), sum(len(v) for v in gets.values()))
        return full

    @staticmethod
    def _resolve_reroutes(full):
        """本机核心未注册 'Reroute' 节点 -> POST 400。
        reroute 是恒等透传，沿链追到真正的源节点后，把它的出边直接重指到源。"""
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        link_src = {}
        for l in links:
            if len(l) >= 5:
                link_src[l[0]] = (str(l[1]), l[2])
        re_ids = set(str(n["id"]) for n in nodes if n.get("type") == "Reroute")
        if not re_ids:
            return full

        def resolved(rid):
            seen, cur, slot = set(), rid, None
            while cur in re_ids and cur not in seen:
                seen.add(cur)
                node = next((n for n in nodes if str(n["id"]) == cur), None)
                if not node:
                    return None
                ins = node.get("inputs", [])
                lnk = ins[0].get("link") if ins else None
                if lnk is None or lnk not in link_src:
                    return None
                slot = link_src[lnk][1]
                cur = link_src[lnk][0]
            if cur in re_ids:
                return None
            return (cur, slot)

        link_by_id = {l[0]: l for l in links if len(l) >= 5}
        cnt = 0
        for rid in list(re_ids):
            r = resolved(rid)
            if not r:
                continue
            src, slot = r
            node = next((n for n in nodes if str(n["id"]) == rid), None)
            if not node:
                continue
            for o in node.get("outputs", []):
                for lid in (o.get("links") or []):
                    if lid in link_by_id:
                        link_by_id[lid][1] = int(src)
                        link_by_id[lid][2] = slot
                        cnt += 1
        full["nodes"] = [n for n in nodes if str(n["id"]) not in re_ids]
        info("SCAILAdapter", "预处理：解析 %d 条 Reroute 边，移除 %d 个 Reroute 节点",
              cnt, len(re_ids))
        return full

    @staticmethod
    def _required_input_names(class_type):
        """返回某 class_type 的必填(required)输入名集合。"""
        oi = SCAILAdapter._object_info()
        info = oi.get(class_type)
        if info:
            return set(info.get("input", {}).get("required", {}).keys())
        return set()

    @staticmethod
    def _delete_unregistered(full):
        """删除 class_type 在本机未注册的节点，并断开其消费方连线。

        关键：只「断开」消费方(把指向已删节点的 input.link 置空)，**绝不级联删除**
        消费节点。原因：消费节点往往仍有 widget 默认值或会被 `_fix_pose_images` 等
        回退逻辑重连(如 pose_images 接缩放驱动帧)。早期实现使用级联删除，导致一个
        未注册节点被删后，其上溯消费链整条被移除(本机实测把 67 节点图删到只剩 2 个)。
        断开而非删除，与已验证可用的 harness `delete_nodes_by_type` 行为一致。"""
        mappings = SCAILAdapter._node_class_mappings()
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        del_ids = set(str(n["id"]) for n in nodes
                      if (n.get("type") or "") not in mappings)
        if not del_ids:
            return full
        # 断开消费方：把指向已删节点输出的 input.link 置空（不删除消费节点）
        for n in nodes:
            if str(n["id"]) in del_ids:
                continue
            for i in n.get("inputs", []):
                lnk = i.get("link")
                if lnk is None:
                    continue
                src = next((l for l in links if l[0] == lnk), None)
                if src and str(src[1]) in del_ids:
                    i["link"] = None
        full["links"] = [l for l in links
                         if str(l[1]) not in del_ids and str(l[3]) not in del_ids]
        full["nodes"] = [n for n in nodes if str(n["id"]) not in del_ids]
        info("SCAILAdapter", "预处理：删除 %d 个未注册节点(仅断开消费方，不级联删除)",
              len(del_ids))
        return full

    @staticmethod
    def _drop_bypassed(full):
        """删除 mode==4（ComfyUI 中『禁用/绕过(bypass)』）的节点，并断开其消费方连线。

        ComfyUI 不执行 mode==4 节点，其输出视为断开。若只在 _convert_full_to_api 里
        用 `mode==4: continue` 跳过该节点、却不清理它的输出连线，则下游消费方仍会引用
        这个不存在的节点 id，导致内联执行报 NodeNotFoundError：
        实测 Tier2 模板里被禁用的 LoraSelectMulti=node66 就触发了『Node 66 not found』
        （其输出 link 62 仍指向 WanAnimatePlus SetLoRAs 的 lora 输入，转换后变成
        [\"66\", slot] 悬空引用）。

        这里与 _delete_unregistered 同理：断开消费方、清理 links、移除节点。
        mode==4 节点的消费方输入断开后由 ComfyUI 当作未连接处理（对应输入默认 None，
        如 SetLoRAs.lora），语义上等价于用户在 UI 里禁用该节点。"""
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        bypass_ids = {str(n["id"]) for n in nodes if (n.get("mode", 0) == 4)}
        if not bypass_ids:
            return full
        for n in nodes:
            if str(n["id"]) in bypass_ids:
                continue
            for i in n.get("inputs", []):
                lnk = i.get("link")
                if lnk is None:
                    continue
                src = next((l for l in links if l[0] == lnk), None)
                if src and str(src[1]) in bypass_ids:
                    i["link"] = None
        full["links"] = [l for l in links
                         if str(l[1]) not in bypass_ids and str(l[3]) not in bypass_ids]
        full["nodes"] = [n for n in nodes if str(n["id"]) not in bypass_ids]
        info("SCAILAdapter", "预处理：丢弃 %d 个禁用(bypass,mode=4)节点: %s",
              len(bypass_ids), sorted(bypass_ids))
        return full

    @staticmethod
    def _delete_dangling_vhs(full):
        """删除 images 输入未连线(缺失/悬空)的 VHS_VideoCombine（例如已被删的姿势预览合成节点）。
        通用判定：只要其 images 输入没有指向一个现存节点输出，即视为悬空丢弃。"""
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        link_src = {l[0]: (str(l[1]), l[2]) for l in links if len(l) >= 5}
        del_ids = set()
        for n in nodes:
            if n.get("type") != "VHS_VideoCombine":
                continue
            ok = False
            for i in n.get("inputs", []):
                if i.get("name") == "images":
                    lnk = i.get("link")
                    if lnk is not None and lnk in link_src:
                        ok = True
                    break
            if not ok:
                del_ids.add(str(n["id"]))
        if del_ids:
            for n in nodes:
                if str(n["id"]) in del_ids:
                    continue
                for i in n.get("inputs", []):
                    lnk = i.get("link")
                    if lnk is None:
                        continue
                    src = next((l for l in links if l[0] == lnk), None)
                    if src and str(src[1]) in del_ids:
                        i["link"] = None
            full["links"] = [l for l in links
                             if str(l[1]) not in del_ids and str(l[3]) not in del_ids]
            full["nodes"] = [n for n in nodes if str(n["id"]) not in del_ids]
            info("SCAILAdapter", "预处理：移除 %d 个悬空 VHS_VideoCombine(images 未连线): %s",
                  len(del_ids), del_ids)
        return full

    @staticmethod
    def _keep_main_chain(full):
        """丢弃与最终视频输出无关的孤立节点（已删姿态/预览子图的残留，如
        SimpleCalculatorKJ、预览 VHS、未用的 ImageConcatMulti 等）。
        做法：从『images 输入仍有效连线』的 VHS_VideoCombine(即主输出)出发，沿输入
        链接向上回溯标记可达节点；其余节点整批删除。保护节点本就在主链上，不受影响。"""
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        link_map = {l[0]: l for l in links if len(l) >= 5}
        starts = []
        for n in nodes:
            if n.get("type") == "VHS_VideoCombine":
                for i in n.get("inputs", []):
                    if i.get("name") == "images" and i.get("link") is not None:
                        starts.append(str(n["id"]))
                        break
        if not starts:
            return full
        reachable = set()
        stack = list(starts)
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            n = next((x for x in nodes if str(x["id"]) == nid), None)
            if not n:
                continue
            for i in n.get("inputs", []):
                lnk = i.get("link")
                if lnk is not None and lnk in link_map:
                    stack.append(str(link_map[lnk][1]))
        del_ids = set(str(n["id"]) for n in nodes if str(n["id"]) not in reachable)
        if not del_ids:
            return full
        for n in nodes:
            if str(n["id"]) in del_ids:
                continue
            for i in n.get("inputs", []):
                lnk = i.get("link")
                if lnk is None:
                    continue
                src = next((l for l in links if l[0] == lnk), None)
                if src and str(src[1]) in del_ids:
                    i["link"] = None
        full["links"] = [l for l in links
                         if str(l[1]) not in del_ids and str(l[3]) not in del_ids]
        full["nodes"] = [n for n in nodes if str(n["id"]) not in del_ids]
        info("SCAILAdapter", "预处理：丢弃 %d 个与输出无关的孤立节点(姿态/预览残留)",
              len(del_ids))
        return full

    @staticmethod
    def _drop_preview_chain(full):
        """丢弃姿势叠加预览链（官方工作流里 ImageConcatMulti->ImageConcatMulti->
        VHS 预览），依赖未装的 ComfyUI-SCAIL-Pose，且不影响主输出。"""
        drop = {"328", "318", "319"}
        nodes = full.get("nodes", [])
        if not any(str(n["id"]) in drop for n in nodes):
            return full
        links = full.get("links", [])
        for n in nodes:
            if str(n["id"]) in drop:
                continue
            for i in n.get("inputs", []):
                lnk = i.get("link")
                if lnk is None:
                    continue
                src = next((l for l in links if l[0] == lnk), None)
                if src and str(src[1]) in drop:
                    i["link"] = None
        full["links"] = [l for l in links
                         if str(l[1]) not in drop and str(l[3]) not in drop]
        full["nodes"] = [n for n in nodes if str(n["id"]) not in drop]
        info("SCAILAdapter", "预处理：丢弃姿势预览链节点 %s", sorted(drop))
        return full

    @staticmethod
    def _convert_full_to_api(full, objinfo):
        """完整 UI 格式 -> API 格式。widgets_values 兼容 dict(key->value) 与
        list 两种格式（新版 ComfyUI 存 dict，旧版存 list）。"""
        nodes_list = full.get("nodes", [])
        links_list = full.get("links", [])
        link_map = {}
        for link in links_list:
            if len(link) >= 5:
                link_map[link[0]] = {"src_node": str(link[1]), "src_slot": link[2],
                                     "dst_node": str(link[3]), "dst_slot": link[4]}
        api = {}
        for node in nodes_list:
            nid = str(node.get("id", ""))
            ct = node.get("type", "")
            if node.get("mode", 0) == 4:
                continue
            linked = set()
            inputs = {}
            for inp in node.get("inputs", []):
                ln = inp.get("link")
                if ln is not None and ln in link_map:
                    li = link_map[ln]
                    inputs[inp["name"]] = [li["src_node"], li["src_slot"]]
                    linked.add(inp["name"])
            wnames = SCAILAdapter._widget_names(ct)
            wv = node.get("widgets_values", [])
            if isinstance(wv, dict):
                for wn in wnames:
                    if wn in wv and wn not in linked:
                        inputs[wn] = wv[wn]
            else:
                wi = 0
                for v in wv:
                    # 跳过节点运行时生成的预览图缓存（如 Schedulerv2 的 sigmas plot base64 图片）
                    # 这类值不是 INPUT_TYPES 定义的 widget，但会被 ComfyUI 存入 widgets_values
                    if isinstance(v, str) and v.startswith("<img src='data:image"):
                        continue
                    # 跳过 ComfyUI 前端自动附加的 control_after_generate（紧跟 seed 后面）
                    # 它不在 INPUT_TYPES 中，但占 widgets_values 一个位置
                    # 不跳过会导致后续 BOOLEAN widget 错位（如 force_offload/add_noise_to_samples）
                    if isinstance(v, str) and v in ("fixed", "increment", "decrement", "randomize") and \
                       wi < len(wnames) and wnames[wi] != "control_after_generate":
                        continue
                    if wi < len(wnames):
                        wn = wnames[wi]
                        if wn not in linked:
                            inputs[wn] = v
                    wi += 1
            api[nid] = {"class_type": ct, "inputs": inputs}
        return api

    @staticmethod
    def _prune_api_missing(api):
        """API 格式：删除 class_type 未注册的节点，并清理引用它们的边。"""
        mappings = SCAILAdapter._node_class_mappings()
        del_ids = {nid for nid, nd in api.items()
                   if nd.get("class_type") not in mappings}
        if not del_ids:
            return api
        for nid, nd in api.items():
            for k, v in list(nd.get("inputs", {}).items()):
                if isinstance(v, list) and len(v) == 2 and v[0] in del_ids:
                    del nd["inputs"][k]
        for nid in del_ids:
            api.pop(nid, None)
        info("SCAILAdapter", "预处理(API)：删除 %d 个未注册节点", len(del_ids))
        return api

    @staticmethod
    def _sanitize_numeric_inputs(api):
        """UI 工作流常把数值输入写成 'disabled' 等非法字符串（例如
        WanAnimatePlus SCAIL_2 Embeds 的 ref_end_percent）。进程内 execute_inline
        侥幸能跑（跳过 /prompt 的严格类型校验），但一旦前置节点修好、执行到该
        节点就会因 float('disabled') 崩溃。这里用 object_info 类型声明，把
        FLOAT/INT 输入里无法转数字的字符串替换为该字段默认值，保证两种执行路径都健壮。"""
        oi = SCAILAdapter._object_info()
        fixed = 0
        for nid, nd in api.items():
            ct = nd.get("class_type")
            spec = oi.get(ct, {}).get("input", {}) if oi else {}
            inputs_decl = {}
            for cat in ("required", "optional"):
                inputs_decl.update(spec.get(cat, {}))
            if not inputs_decl:
                continue
            for k, v in list(nd.get("inputs", {}).items()):
                if isinstance(v, list):
                    continue  # 连线，不动
                decl = inputs_decl.get(k)
                if not (isinstance(decl, list) and decl):
                    continue
                typ = decl[0]
                if typ in ("FLOAT", "INT") and isinstance(v, str):
                    try:
                        if typ == "INT":
                            int(v)
                        else:
                            float(v)
                    except (ValueError, TypeError):
                        default = 0.0
                        try:
                            if isinstance(decl[1], dict):
                                default = decl[1].get("default", 0.0)
                        except Exception:
                            pass
                        default = int(default) if typ == "INT" else float(default)
                        nd["inputs"][k] = default
                        fixed += 1
                        warn("SCAILAdapter", "sanitize: 节点%s(%s) 输入 '%s' 非法值 %r → 默认值 %s",
                             nid, ct, k, v, default)
        if fixed:
            info("SCAILAdapter", "sanitize: 修正 %d 个非法数值输入", fixed)
        return api

    @staticmethod
    def _workflow_has_distill_lora(wf):
        """判断工作流是否已挂载『步数蒸馏 LoRA』(lightx2v step_distill 等)。
        蒸馏 LoRA 为 4 步设计，强行提高步数反而会因 schedule 错配产生结构性
        伪影(即用户所说的『崩坏』)，故一旦检测到蒸馏 LoRA 就放行原生快速步数。

        关键修正：本机用户实际工作流用 WanAnimatePlus LoraSelectMulti 挂载蒸馏
        LoRA，字段名是 lora_0/lora_1/...（不是 lora/lora_name）。旧实现只扫
        lora/lora_name/model_name 三个固定字段名 → 漏检 → 放行失败 → 被
        _enforce_min_sampling_steps 强制 25 步 → 崩坏依旧。现改为扫描『所有输入
        值』是否含 step-distill 标识("distill")，覆盖单/多 LoRA 选择节点的任意字段。"""
        for nd in wf.values():
            if not isinstance(nd, dict):
                continue
            inp = nd.get("inputs", {})
            if not isinstance(inp, dict):
                continue
            for fk, v in inp.items():
                # 任意字符串值带 step-distill 标识即命中（lora/lora_0..N/lora_name 等）
                if isinstance(v, str) and "distill" in v.lower():
                    return True
                # 含 'lora' 的字段且值为 distill LoRA 文件名（双保险）
                if isinstance(fk, str) and "lora" in fk.lower() \
                        and isinstance(v, str) and "distill" in v.lower():
                    return True
        return False

    @staticmethod
    def _enforce_min_sampling_steps(wf, min_steps=20, target=25):
        """防御性：SCAIL-2 路线用的 wan2.1_14B_SCAIL_2 是基座模型(非步数蒸馏)，
        采样步数过低会严重欠去噪 → 生成内容高频细节丢失、整体模糊。
        实测 4 步清晰度≈14、25 步≈57(拉普拉斯方差)。
        SCAIL-2 的步数可能挂在采样器(WanAnimatePlus SamplerSettings)或调度器
        (WanVideoSchedulerv2)上，故按输入键名精确匹配 'steps'(不靠 class_type，
        否则会漏掉含 Scheduler 的调度器节点)。任何 SCAIL 节点的 steps 低于阈值一律
        提到 target，避免模板/UI 旧图再次掉回低步数导致模糊。

        关键例外：若工作流已挂载步数蒸馏 LoRA(lightx2v/step_distill)，则 4 步即正确，
        必须放行快速步数——否则蒸馏 LoRA 在错误高步数下会产生崩坏伪影。
        仅在 SCAILAdapter / AnimatePlusSCAILAdapter 调用，不波及骨骼路线(蒸馏模型 4-8 步正确)。"""
        if SCAILAdapter._workflow_has_distill_lora(wf):
            # 蒸馏 LoRA 路线：不再只是『放行』，而是主动把全部 steps 节点钉到 4 步。
            # 原因：旧模板/UI 缓存可能仍残留 25 步（用户直跑工作流曾被强制 25），
            # 纯放行会保留那个 25 → 崩坏依旧。强制 4 步给两条路线(官方子流程 /
            # WanAnimatePlus)都上安全网，与好片 00017(4 步)一致。
            fixed = 0
            for nid, nd in wf.items():
                if not isinstance(nd, dict):
                    continue
                inp = nd.get("inputs")
                if isinstance(inp, dict) and isinstance(inp.get("steps"), (int, float)) \
                        and not isinstance(inp.get("steps"), bool) and int(inp["steps"]) != 4:
                    inp["steps"] = 4
                    fixed += 1
            info("SCAILAdapter", "步数防御: 蒸馏 LoRA 路线 → 强制 %d 个 steps 节点=4(快速)", fixed)
            return wf
        fixed = 0
        for nid, nd in wf.items():
            if not isinstance(nd, dict):
                continue
            ct = nd.get("class_type", "")
            inp = nd.setdefault("inputs", {})
            for k, v in list(inp.items()):
                if k.lower() == "steps" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    if int(v) < min_steps:
                        inp[k] = target
                        fixed += 1
                        warn("SCAILAdapter", "步数防御: node%s(%s) steps=%s → %s", nid, ct, v, target)
        if fixed:
            info("SCAILAdapter", "步数防御: 修正 %d 个低步数采样节点 → %s", fixed, target)
        return wf

    @staticmethod
    def _fix_model_names(api):
        """通用：凡 model_name / lora / lora_name 当前值不在本机该节点可用文件列表里，
        自动选最相近的可用文件（difflib 模糊匹配, sim>0.6）。
        选项来源直接用 /object_info 里『该 class_type + 该字段』的 COMBO 列表，
        避免 folder_paths.get_filename_list 在独立进程/代理环境下返回不准或触发重导入。
        仅处理模型/LoRA 加载器，绝不碰扩散模型加载器的 'model' 字段。"""
        import difflib
        oi = SCAILAdapter._object_info()
        FIXABLE = ("model_name", "lora", "lora_name")
        for nid, nd in api.items():
            ct = nd.get("class_type", "")
            oinfo = oi.get(ct)
            if not oinfo:
                continue
            meta = {}
            for cat in ("required", "optional"):
                meta.update(oinfo.get("input", {}).get(cat, {}))
            for field in FIXABLE:
                if field not in nd.get("inputs", {}):
                    continue
                cfg = meta.get(field)
                if not (isinstance(cfg, list) and cfg and isinstance(cfg[0], list)):
                    continue
                opts = cfg[0]  # 该字段在本机的全部可用文件名
                cur = nd["inputs"][field]
                if not isinstance(cur, str) or not cur:
                    continue
                if cur in opts:
                    continue
                # 文本编码器家族感知：WanVideoTextEncode* 节点只要 Wan 的 umT5，
                # 且显式拒绝 fp8_scaled（节点加载时会 raise）。避免 umt5 类名字
                # 因共享 'fp8_e4m3fn' 后缀被 difflib 错配到 SDXL 的 t5xxl
                # （t5xxl 仅 block0 有 relative_attention_bias，转换后缺
                #  blocks.N.pos_embedding，umt5_xxl 模型加载即 KeyError）。
                cands = opts
                if ct.lower().startswith("wanvideotextencode"):
                    cl = cur.lower()
                    if "umt5" in cl:
                        pref = [o for o in opts
                                if "umt5" in o.lower() and "scaled" not in o.lower()]
                        if pref:
                            cands = pref
                    elif "t5xxl" in cl or "t5-xxl" in cl:
                        pref = [o for o in opts if "t5xxl" in o.lower()]
                        if pref:
                            cands = pref
                best, bestr = None, 0.0
                for o in cands:
                    r = difflib.SequenceMatcher(None, cur.lower(), o.lower()).ratio()
                    if r > bestr:
                        bestr, best = r, o
                if best and bestr > 0.6:
                    nd["inputs"][field] = best
                    info("SCAILAdapter", "模型名模糊匹配 %s.%s: %r -> %r (sim=%.2f)",
                         nid, field, cur, best, bestr)

    @staticmethod
    def _fix_pose_images(api, driving_video_node=None):
        """若 ComfyUI-SCAIL-Pose 未装，pose_images 源节点(检测/渲染)已被删 ->
        WanVideoAddSCAILPoseEmbeds.pose_images 悬空。回退：接【完整驱动视频】(VHS_LoadVideo)。

        关键修复（2026-07-28 日志崩溃）：WanVideoAddSCAILPoseEmbeds 的 pose_images
        契约是 "Pose images for the entire video"，VAE 编码后 pose_latent 帧数必须覆盖
        全部生成帧。采样时 nodes_sampler.py:1416 会按 context_window
        (当前上下文的各潜在帧索引，如 81 帧 -> [0..20]) 取
        scail_data["pose_latent"][:, context_window]。若 pose_latent 仅 1 帧
        (旧回退接单帧缩放节点 ImageResize/ImageScale 所致)，索引 >=1 即越界 ->
        CUDA device-side assert (index out of bounds)，采样第 0 步即崩。

        因此这里必须接能产出【全部帧】的节点(VHS_LoadVideo)，不能用单帧缩放节点。
        装上 SCAIL-Pose 后官方姿态分支保留，此回退不触发。"""
        for nid, nd in api.items():
            if nd["class_type"] != "WanVideoAddSCAILPoseEmbeds":
                continue
            if "pose_images" in nd["inputs"]:
                continue
            src = None
            # 1) 优先用完整驱动视频节点(全部帧) —— 关键：保证 pose_latent 帧数覆盖生成段
            if driving_video_node and driving_video_node in api:
                src = driving_video_node
            else:
                for k, v in api.items():
                    if v["class_type"] == "VHS_LoadVideo":
                        src = k
                        break
            # 2) 兜底：实在没有视频节点，用单帧缩放节点(仅结构性占位，长段仍可能越界)
            if not src:
                for k, v in api.items():
                    if "ImageResize" in v["class_type"] or v["class_type"] in ("ImageScale", "ImageResizeKJv2"):
                        src = k
                        break
            if src:
                nd["inputs"]["pose_images"] = [src, 0]
                warn("SCAILAdapter",
                     "pose_images 悬空，回退接完整驱动视频节点 %s（非官方骨架姿态；建议安装 ComfyUI-SCAIL-Pose 以获真骨架）",
                     src)
        return api

    @staticmethod
    def _pose_weights_available():
        """官方 NLF 姿态分支需要的权重（nlf 模型 + detection 检测器）是否本机齐全。

        注意：本机 folder_paths 未注册 'nlf' / 'onnx' 等类型（get_folder_paths 会抛
        KeyError），所以不能直接用 have("nlf")。改为以 models 根目录为基准直接扫
        文件系统：detection 类型已注册可反推 models 根，nlf 目录同理。"""
        import folder_paths as _fp
        import os
        base = getattr(_fp, "models_dir", None)
        if not base:
            try:
                det = _fp.get_folder_paths("detection")
                if det:
                    base = os.path.dirname(det[0])
            except Exception:
                pass
        if not base:
            return False

        def has_files(sub, exts):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                return False
            try:
                return any(f.name.lower().endswith(exts) for f in os.scandir(d))
            except Exception:
                return False

        nlf_ok = has_files("nlf", (".safetensors", ".torchscript", ".pt", ".bin"))
        det_ok = has_files("detection", (".onnx",))
        return nlf_ok and det_ok

    def _official_pose_runnable(self, full):
        """官方 NLF 姿态分支在本机能否真正跑通：
        1) nlf 模型 + onnx 检测器权重齐全（检测权重在 models/detection/）；
        2) 工作流喂给 RenderNLFPoses 的输入名当前包版本 INPUT_TYPES 全部接受
           （如旧 width/height 已被 render_width/render_height 取代；任何未来改名同理）。
        任一不满足 -> 返回 False，调用方应丢弃该子图、回退驱动帧。"""
        if not self._pose_weights_available():
            return False
        oi = self._object_info()
        pose_info = oi.get("RenderNLFPoses")
        if not pose_info:
            return False  # 无法判定 -> 视为不兼容，回退驱动帧
        accepted = set()
        for cat in ("required", "optional"):
            accepted |= set(pose_info.get("input", {}).get(cat, {}).keys())
        for n in full.get("nodes", []):
            if n.get("type") != "RenderNLFPoses":
                continue
            fed = {i.get("name") for i in n.get("inputs", [])
                     if isinstance(i, dict)}
            # 工作流喂给 RenderNLFPoses 的输入名若当前包 INPUT_TYPES 不接受
            # （如旧 width/height，或任何未来改名），即视为不兼容 -> 回退驱动帧
            if fed - accepted:
                return False
        return True

    @staticmethod
    def _drop_pose_branch(full):
        """丢弃官方 NLF 姿态渲染子图（权重缺失 / 版本不兼容时），
        使 WanVideoAddSCAILPoseEmbeds.pose_images 悬空，交给 _fix_pose_images 回退驱动帧。"""
        drop_types = {"NLFPredict", "NLFPredictPoses", "DownloadAndLoadNLFModel", "NLFModelLoader",
                      "RenderNLFPoses", "PoseDetectionVitPoseToDWPose",
                      "OnnxDetectionModelLoader"}
        nodes = full.get("nodes", [])
        links = full.get("links", [])
        del_ids = set(str(n["id"]) for n in nodes
                      if (n.get("type") or "") in drop_types)
        if not del_ids:
            return full
        for n in nodes:
            if str(n["id"]) in del_ids:
                continue
            for i in n.get("inputs", []):
                lnk = i.get("link")
                if lnk is None:
                    continue
                src = next((l for l in links if l[0] == lnk), None)
                if src and str(src[1]) in del_ids:
                    i["link"] = None
        full["links"] = [l for l in links
                         if str(l[1]) not in del_ids and str(l[3]) not in del_ids]
        full["nodes"] = [n for n in nodes if str(n["id"]) not in del_ids]
        info("SCAILAdapter", "预处理：丢弃 NLF 姿态子图 %d 节点（权重缺失/版本不兼容），回退驱动帧",
              len(del_ids))
        return full

    def prepare_workflow(self, workflow_raw):
        """把官方 SCAIL UI 工作流整理成干净、可提交的 API 工作流。

        处理本机差异：KJ Set/Get、Reroute、未装插件(Note/SCAIL-Pose)、
        widgets_values 新版 dict 格式、模型/LoRA 路径前缀、尺寸常量。
        返回 API 格式 dict（供 discover_nodes / modify_workflow_for_segment 使用）。
        """
        if not isinstance(workflow_raw, dict):
            warn("SCAILAdapter", "prepare_workflow: 输入非 dict")
            return {}
        # 已是 API 格式？
        first = next(iter(workflow_raw), None)
        if first is not None and isinstance(workflow_raw[first], dict) \
                and ("class_type" in workflow_raw[first] or "type" in workflow_raw[first]):
            api = self._prune_api_missing(dict(workflow_raw))
            self._sanitize_numeric_inputs(api)
            self._fix_model_names(api)
            self._fix_pose_images(api)
            return api
        # UI 完整格式：手术 + 转换
        full = json.loads(json.dumps(workflow_raw))
        full = self._rewire_setget(full)
        full = self._resolve_reroutes(full)
        full = self._delete_unregistered(full)
        if not self._official_pose_runnable(full):
            full = self._drop_pose_branch(full)
        full = self._delete_dangling_vhs(full)
        full = self._keep_main_chain(full)
        full = self._drop_preview_chain(full)
        # 官方 NLF 姿态分支在本机跑不通（权重缺失 / 当前包版本不兼容
        # 如 RenderNLFPoses 已移除 width/height）时，丢弃该子图，
        # 回退到「驱动帧作 pose_images」的可靠路径，避免执行 400。
        if not self._official_pose_runnable(full):
            full = self._drop_pose_branch(full)
        full = self._drop_bypassed(full)
        mappings = self._node_class_mappings()
        api = self._convert_full_to_api(full, mappings)
        self._sanitize_numeric_inputs(api)
        missing = {nd["class_type"] for nd in api.values()
                   if nd["class_type"] not in mappings}
        if missing:
            warn("SCAILAdapter", "prepare_workflow: 仍有未注册 class_type: %s", missing)
        self._fix_model_names(api)
        self._fix_pose_images(api)
        info("SCAILAdapter", "prepare_workflow: 整理后 %d 个节点", len(api))
        return api
