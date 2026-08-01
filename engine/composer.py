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
import os

from .runner import YunjiiSegmentRunner
from .stitcher import YunjiiSegmentStitcher, _build_output_ui, XFADE_NAME_MAP
from .types import (
    SEGMENT_MODE_ONE_SHOT, STITCH_HARD_CUT, STITCH_SEAMLESS, STITCH_SEAMLESS_BLEND,
    STITCH_TRANSITION, STITCH_LABELS, STITCH_LABEL_TO_VALUE,
)
from .debug_log import node_start, node_end, node_error, info, warn

logger = logging.getLogger(__name__)

# 拼接模式下拉 = 中文标签（内部比较仍用英文值；旧 saved 英文值经 STITCH_LABEL_TO_VALUE 归一）
_STITCH_MODES = [label for _, label in STITCH_LABELS]


def _coerce_float(v, default):
    """把 widget 值稳妥转成 float；空串/None/非法值回落到 default。
    用于兼容旧工作流把数字 widget 存成 '' 的情况（避免校验期崩溃）。"""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_int(v, default):
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


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
                    {"default": "ffmpeg转场", "tooltip": "硬切=直接拼接(一镜到底会被防呆回退重叠混合); 交叉淡化=像素级平滑过渡; 自动=根据内容选择(一镜到底落重叠混合); 无缝一镜到底(零转场)=硬切去重零转场; 无缝一镜到底(重叠混合)=接缝短窗交叉溶解软化跳变; ffmpeg转场=视频级高级转场(选「转场类型」生效，一镜到底可用作0.5s平滑接缝)"},
                ),
                "淡化帧数": ("INT", {"default": 8, "min": 2, "max": 30, "step": 1,
                    "tooltip": "交叉淡化过渡帧数（仅 交叉淡化 模式生效）"}),
                "转场类型": (list(XFADE_NAME_MAP.keys()), {"default": "淡入淡出",
                    "tooltip": "ffmpeg xfade 视频级转场类型（仅 ffmpeg转场 拼接模式生效）。自动=固定淡入淡出；随机=每个接缝随机抽一种；推荐：淡入淡出"}),
                "转场时长": ("STRING", {"default": "0.5",
                    "tooltip": "转场时长(秒，填数字)，需小于每段时长，否则自动回退交叉淡化。推荐 0.5s"}),
                "输出文件名": ("STRING", {"default": "yunjii_v2v", "tooltip": "输出文件名前缀"}),
            },
            "optional": {
                "视频路径": ("STRING", {"default": "", "tooltip": "参考视频路径（从运动分析节点连线传入）"}),
                "参考图": ("IMAGE", {"tooltip": "参考图（从LoadImage节点连线传入），优先使用"}),
                "姿态图": ("IMAGE", {"tooltip": "姿态引导图（从VideoPoseExtractor节点连线传入）"}),
                "人物参考图": ("STRING", {"default": "", "tooltip": "参考图文件名（input目录下），连线传入参考图时此项可忽略"}),
                "起始段": ("STRING", {"default": "0", "tooltip": "从第几段开始生成（填数字，默认0=从头）。兼容旧工作流空值"}),
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
                起始段=0, 音频源="", 效果模块="", ComfyUI地址="127.0.0.1:8188",
                转场类型="淡入淡出", 转场时长=0.5):
        node_start("Imitator", 执行模式=执行模式, 生成后端=生成后端, 拼接模式=拼接模式, 效果模块=效果模块 or "(空)")
        # —— 健壮性：widget 改为 STRING 后，旧工作流可能残留空串/非法值，这里兜底转换 ——
        转场时长 = _coerce_float(转场时长, 0.5)
        起始段 = _coerce_int(起始段, 0)
        if 转场类型 not in XFADE_NAME_MAP:
            info("Imitator", f"转场类型'{转场类型}'非法，回退默认'淡入淡出'")
            转场类型 = "淡入淡出"

        # —— 拼接模式中文标签 → 英文值归一（兼容旧 saved 英文值）——
        _raw_mode = 拼接模式
        拼接模式 = STITCH_LABEL_TO_VALUE.get(拼接模式, 拼接模式)
        if _raw_mode != 拼接模式:
            info("Imitator", "拼接模式归一: '%s' → '%s'", _raw_mode, 拼接模式)

        # —— 一镜到底：生成模式即「零转场连续长镜头」，拼接必须走 seamless(去重硬切) ——
        # 否则 auto 会在段边界连续(差异<30)时误选交叉淡化，产生可见溶解转场，违背一镜到底本意。
        # 这里读取段落计划的生成模式，强制覆盖拼接模式（用户误选交叉淡化也无效）。
        try:
            _plan_mode = ""
            _plan_single_pass = False
            _plan_continuity = ""
            _plan_precision = "fp8"
            _plan_raw = 段落计划.strip()
            if _plan_raw:
                _pd = json.loads(_plan_raw)
                _plan_mode = _pd.get("mode", "")
                _plan_single_pass = bool(_pd.get("single_pass", False))
                _plan_continuity = _pd.get("continuity_strategy", "")
                _plan_precision = _pd.get("model_precision", "fp8")
        except Exception:
            _plan_mode = ""
            _plan_single_pass = False
            _plan_continuity = ""
            _plan_precision = "fp8"
        # —— 一镜到底：拼接模式防呆（仅拦截真正破坏连贯的「硬切」）——
        # 一镜到底=连续长镜头；硬切会暴露段边界跳变，故强制回退 重叠混合。
        # ffmpeg转场(xfade 真·交叉溶解) 与 交叉淡化 本质都是「平滑过渡」，恰是一镜到底想要的丝滑，
        # 故放行（用户选 ffmpeg转场+淡入淡出 即恢复此前最喜欢的 0.5s 平滑接缝）。
        # 自动(auto) 已落 seamless_blend；零转场/重叠混合 均为用户显式基线，保留。
        if _plan_mode == SEGMENT_MODE_ONE_SHOT and 拼接模式 == STITCH_HARD_CUT:
            info("Imitator", "生成模式=一镜到底，硬切会暴露段边界跳变，强制回退「无缝一镜到底(重叠混合)」")
            拼接模式 = STITCH_SEAMLESS_BLEND

        # —— 单一效果模块，份传给两侧（本节点核心价值）——
        effects = 效果模块 or ""

        # 1) 生成相位：调用链式执行引擎，效果模块作用于每段 prompt/params
        runner = YunjiiSegmentRunner()
        try:
            执行结果, 执行日志, 完成状态 = runner.run(
                段落计划, 工作流模板, 执行模式, 最大重试,
                生成后端=生成后端, 视频路径=视频路径, 参考图=参考图, 姿态图=姿态图,
                人物参考图=人物参考图, 起始段=起始段, 效果模块=effects, ComfyUI地址=ComfyUI地址,
                连贯策略=_plan_continuity, 模型精度=_plan_precision,
            )
        except Exception as _exc:
            import traceback as _tb
            _detail = "".join(_tb.format_exception_only(type(_exc), _exc)).strip()
            _stack = _tb.format_exc()
            node_error("Imitator", "生成阶段异常: %s" % _detail)
            info("Imitator", "异常堆栈:\n%s", _stack)
            node_end("Imitator", "生成失败(异常)")
            return ("", f"{执行日志}\n[异常] {_detail}", False)

        # 仅规划：runner 已返回计划摘要，直接短路，不做拼接
        if 执行模式 == "仅规划":
            info("Imitator", "仅规划模式，跳过生成与拼接")
            node_end("Imitator", "仅规划")
            return ("", 执行日志, 完成状态)

        if not 完成状态:
            # 透传 runner 返回的真实错误信息（执行结果里含具体原因），不再笼统吞掉
            _err_msg = (执行结果 or "未知原因").strip()
            node_error("Imitator", "生成阶段失败: %s" % _err_msg)
            node_end("Imitator", "生成失败")
            return ("", f"{执行日志}\n[生成失败] {_err_msg}", False)

        # 一镜到底单次超长(方案C)：单段即成片，无需拼接，直接返回该段视频
        if _plan_single_pass:
            try:
                _res = json.loads(执行结果)
                _segs = _res.get("segments", [])
                if _segs and _segs[0].get("video_path"):
                    _final = _segs[0]["video_path"]
                    # 方案C 单遍跳过 stitcher，需在此补回音频混流（与多段路径一致）。
                    # 优先「音频源」，为空则回退驱动「视频路径」（二者均带原参考视频音轨）。
                    _audio_src = 音频源 or ""
                    if not _audio_src and 视频路径 and str(视频路径).lower().endswith(
                            (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
                        _audio_src = 视频路径
                    if _audio_src and os.path.isfile(_audio_src):
                        try:
                            _stitcher = YunjiiSegmentStitcher()
                            _final = _stitcher._add_audio(_final, _audio_src, 输出文件名,
                                                          _res.get("run_id", ""))
                            info("Imitator", "一镜到底单遍已混入原始音频: %s", _final)
                        except Exception as e:
                            warn("Imitator", "单遍音频混流失败(不影响成片): %s", e)
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
            转场类型=转场类型, 转场时长=转场时长,
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
