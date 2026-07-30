"""
Yunjii 长视频完美模仿 —— 贯穿节点（Composer）

把「分段规划 → 链式执行(生成) → 无缝拼接」三步合一为单个节点，
**单一 `效果模块` 入参同时份传给生成相位与拼接相位**，
从根上消除"runner 与 stitcher 各自吃一份 效果模块，填不一致会割裂效果"的隐患。

设计要点：
- 沿用现有 YunjiiSegmentRunner / YunjiiSegmentStitcher 的真实逻辑，本节点只做编排，
  不重复实现生成/拼接，零回归。
- 仅规划模式下短路：不进入生成/拼接，直接返回规划摘要。
- 执行/续跑：先跑 runner.run()，成功后再跑 stitcher.stitch()；
  任一阶段失败即提前返回，不掩盖错误。
- 效果模块"一处定义、两侧复用"是本节点最核心的价值。
"""
import logging
import json

from .runner import YunjiiSegmentRunner
from .stitcher import YunjiiSegmentStitcher, _build_output_ui
from .types import SEGMENT_MODE_ONE_SHOT, STITCH_SEAMLESS
from .debug_log import node_start, node_end, node_error, info, warn

logger = logging.getLogger(__name__)

# 拼接模式与 stitcher 保持一致（避免与 stitcher 硬编码值漂移）
_STITCH_MODES = ["hard_cut", "cross_dissolve", "auto", "seamless"]


class YunjiiVideoImitator:
    CATEGORY = "Yunjii/Video/Engine"
    FUNCTION = "imitate"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("最终视频路径", "拼接报告", "完成状态")
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
                "拼接模式": (
                    _STITCH_MODES,
                    {"default": "auto", "tooltip": "硬切=直接拼接; 交叉淡化=平滑过渡; 自动=根据内容选择"},
                ),
                "淡化帧数": ("INT", {"default": 8, "min": 2, "max": 30, "step": 1,
                    "tooltip": "交叉淡化过渡帧数"}),
                "输出文件名": ("STRING", {"default": "yunjii_v2v", "tooltip": "输出文件名前缀"}),
            },
            "optional": {
                "视频路径": ("STRING", {"default": "", "tooltip": "参考视频路径（从运动分析节点连线传入）"}),
                "参考图": ("IMAGE", {"tooltip": "参考图（从LoadImage节点连线传入），优先使用"}),
                "姿态图": ("IMAGE", {"tooltip": "姿态引导图（从VideoPoseExtractor节点连线传入）"}),
                "人物参考图": ("STRING", {"default": "", "tooltip": "参考图文件名（input目录下），连线传入参考图时此项可忽略"}),
                "起始段": ("INT", {"default": 0, "min": 0, "max": 100}),
                "音频源": ("STRING", {"default": "", "tooltip": "原始参考视频路径，用于提取音频"}),
                # —— 核心：单一效果模块，同时作用于生成相位与拼接相位 ——
                "效果模块": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "可选效果管线模块列表(JSON数组或逗号分隔)，如 [\"mimic\",\"enhance\",\"cinematic\"]。"
                                "为空=不启用任何效果，行为与现状完全一致。本节点会将其同时份传给生成与拼接两侧。"}),
                "ComfyUI地址": ("STRING", {"default": "127.0.0.1:8188", "tooltip": "ComfyUI 服务地址（生成阶段使用）"}),
            },
        }

    def imitate(self, 段落计划, 工作流模板, 执行模式, 最大重试, 拼接模式, 淡化帧数, 输出文件名,
                生成后端="骨骼路线(WanVideo)", 视频路径="", 参考图=None, 姿态图=None, 人物参考图="",
                起始段=0, 音频源="", 效果模块="", ComfyUI地址="127.0.0.1:8188"):
        node_start("Imitator", 执行模式=执行模式, 生成后端=生成后端, 拼接模式=拼接模式, 效果模块=效果模块 or "(空)")

        # —— 一镜到底：生成模式即「零转场连续长镜头」，拼接必须走 seamless(去重硬切) ——
        # 否则 auto 会在段边界连续(差异<30)时误选交叉淡化，产生可见溶解转场，违背一镜到底本意。
        # 这里读取段落计划的生成模式，强制覆盖拼接模式（用户误选交叉淡化也无效）。
        try:
            _plan_mode = ""
            _plan_single_pass = False
            _plan_raw = 段落计划.strip()
            if _plan_raw:
                _pd = json.loads(_plan_raw)
                _plan_mode = _pd.get("mode", "")
                _plan_single_pass = bool(_pd.get("single_pass", False))
        except Exception:
            _plan_mode = ""
            _plan_single_pass = False
        if _plan_mode == SEGMENT_MODE_ONE_SHOT and 拼接模式 != STITCH_SEAMLESS:
            info("Imitator", "生成模式=一镜到底，强制拼接模式=无缝一镜到底(seamless)，忽略用户所选=%s", 拼接模式)
            warn("Imitator", "一镜到底模式已强制使用「无缝一镜到底」拼接（去重硬切，零转场），忽略拼接模式=%s", 拼接模式)
            拼接模式 = STITCH_SEAMLESS

        # —— 单一效果模块，份传给两侧（本节点核心价值）——
        effects = 效果模块 or ""

        # 1) 生成相位：调用链式执行引擎，效果模块作用于每段 prompt/params
        runner = YunjiiSegmentRunner()
        执行结果, 执行日志, 完成状态 = runner.run(
            段落计划, 工作流模板, 执行模式, 最大重试,
            生成后端=生成后端, 视频路径=视频路径, 参考图=参考图, 姿态图=姿态图,
            人物参考图=人物参考图, 起始段=起始段, 效果模块=effects, ComfyUI地址=ComfyUI地址,
        )

        # 仅规划：runner 已返回计划摘要，直接短路，不做拼接
        if 执行模式 == "仅规划":
            info("Imitator", "仅规划模式，跳过生成与拼接")
            node_end("Imitator", "仅规划")
            return ("", 执行日志, 完成状态)

        if not 完成状态:
            node_error("Imitator", "生成阶段失败，终止")
            node_end("Imitator", "生成失败")
            return ("", 执行日志, False)

        # 一镜到底单次超长(方案C)：单段即成片，无需拼接，直接返回该段视频
        if _plan_single_pass:
            try:
                _res = json.loads(执行结果)
                _segs = _res.get("segments", [])
                if _segs and _segs[0].get("video_path"):
                    _final = _segs[0]["video_path"]
                    info("Imitator", "一镜到底单次超长成片(方案C): %s (单次生成无需拼接)", _final)
                    node_end("Imitator", "单次超长完成")
                    return {"ui": _build_output_ui(_final),
                            "result": (_final, f"{执行日志}\n---\n✅ 一镜到底单次超长生成，单段即成片，无需拼接", True)}
            except Exception as e:
                warn("Imitator", "单次超长结果解析失败，回退拼接: %s", e)

        # 2) 拼接相位：调用无缝拼接，效果模块作用于成片（超分/插帧/调色等）
        stitcher = YunjiiSegmentStitcher()
        最终视频路径, 拼接报告 = stitcher.stitch(
            执行结果, 拼接模式, 淡化帧数, 输出文件名,
            音频源=音频源, 效果模块=effects,
        )

        if not 最终视频路径:
            node_error("Imitator", "拼接阶段未产出视频")
            node_end("Imitator", "拼接失败")
            return ("", f"{执行日志}\n---\n{拼接报告}", False)

        # 合并两侧日志，便于一次查看全链路
        combined = f"{执行日志}\n---\n{拼接报告}" if 执行日志 else 拼接报告
        info("Imitator", "完美模仿完成: %s", 最终视频路径)
        node_end("Imitator", "完成")
        return {
            "ui": _build_output_ui(最终视频路径),
            "result": (最终视频路径, combined, True),
        }
