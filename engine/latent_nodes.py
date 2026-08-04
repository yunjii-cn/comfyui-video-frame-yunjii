"""Yunjii latent I/O 节点：潜空间拼接方案的存取基础设施。

- YunjiiSaveLatent：插在 WanVideoSamplerv2(sampler) → WanVideoDecode 之间，
  透传 LATENT 零改动生成，副作用把 latent dict 落盘(.pt，含 samples/end_image 等全部键)，
  供后续潜空间拼接的合并/解码使用。
- YunjiiLoadLatent：解码子工作流里加载落盘的合并 latent，返回 LATENT 给 WanVideoDecode。

用 torch.save/load（而非 safetensors）是因为 WanVideoDecode 的 latent dict 除 "samples" 张量外，
还含 end_image/has_ref/drop_last/is_looped 等标量/张量键，需原样保留以保证解码一致。
"""
import os
import logging

import torch

logger = logging.getLogger(__name__)


class YunjiiSaveLatent:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "save"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "来自 WanVideoSamplerv2 的采样 latent（透传）"}),
                "save_path": ("STRING", {"default": "", "tooltip": "latent 落盘路径(.pt)；为空则不保存"}),
            }
        }

    def save(self, samples, save_path):
        if save_path:
            try:
                parent = os.path.dirname(os.path.abspath(save_path))
                os.makedirs(parent, exist_ok=True)
                # 全部键原样落盘（张量转 CPU 省显存/体积），透传原 samples 给 decode
                cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                       for k, v in samples.items()}
                torch.save(cpu, save_path)
                logger.info("[YunjiiSaveLatent] 已保存 latent: %s (%d 键)", save_path, len(cpu))
            except Exception as e:
                logger.warning("[YunjiiSaveLatent] 保存失败(不影响生成): %s", e)
        return (samples,)


class YunjiiLoadLatent:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "load"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "load_path": ("STRING", {"default": "", "tooltip": "latent 文件路径(.pt)，由 YunjiiSaveLatent 生成"}),
            }
        }

    def load(self, load_path):
        if not load_path or not os.path.isfile(load_path):
            raise FileNotFoundError(f"latent 文件不存在: {load_path}")
        d = torch.load(load_path, map_location="cpu", weights_only=False)
        if not isinstance(d, dict) or "samples" not in d:
            raise ValueError(f"latent 文件格式错误(缺 samples 键): {load_path}")
        logger.info("[YunjiiLoadLatent] 已加载 latent: %s", load_path)
        return (d,)
