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
from .stitcher import YunjiiSegmentStitcher, _build_output_ui, _make_cover, XFADE_NAME_MAP
from .frontend_registry import register_video_to_history
from .types import (
    SEGMENT_MODE_ONE_SHOT, STITCH_HARD_CUT, STITCH_SEAMLESS, STITCH_SEAMLESS_BLEND,
    STITCH_FRAME_ANCHOR, STITCH_LATENT_BLEND, STITCH_CROSS_DISSOLVE, STITCH_TRANSITION, STITCH_AUTO,
    STITCH_LABELS, STITCH_LABEL_TO_VALUE, STITCH_DEFAULT,
    CONTINUITY_MULTI_SEG, CONTINUITY_SINGLE_PASS, CONTINUITY_WARM_START,
    SEAMLESS_PLAN_A, SEAMLESS_PLAN_B, SEAMLESS_PLAN_C, SEAMLESS_PLAN_AUTO,
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
    # 末尾追加 IMAGE(封面帧)：与 Stitcher 一致，成为标准输出节点。IMAGE 置于末尾，
    # 旧连线(视频路径/报告/完成状态)不受影响。
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("最终视频路径", "拼接报告", "完成状态", "封面帧")
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
                    {"default": "自动(跟随方案最优)", "tooltip": "默认「自动」即可：分段规划选完后自动跟随后端选最优——SCAIL-2 路线→真·零转场一镜到底；骨骼路线→交叉淡化平滑过渡。无需再选第二次。其余选项为手动覆盖：无缝一镜到底(零转场)=硬切去重零转场; 交叉淡化=像素级平滑过渡[转场]; 潜空间交叉淡化=latent 层转场[转场]; 硬切=直接拼接; ffmpeg转场=视频级高级转场(选「转场类型」生效，一镜到底可用作0.5s平滑接缝)"},
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
        # 旧工作流若残留空串/非法值，兜底为「自动(跟随方案最优)」，避免静默回退到首个可用值。
        拼接模式 = STITCH_LABEL_TO_VALUE.get(拼接模式, STITCH_DEFAULT)
        if 拼接模式 == STITCH_LABEL_TO_VALUE.get(_raw_mode, _raw_mode) and _raw_mode not in ("", None):
            # 命中了中文标签或旧英文值中的某一个，记录归一轨迹
            info("Imitator", "拼接模式归一: '%s' → '%s'", _raw_mode, 拼接模式)
        elif _raw_mode in ("", None):
            info("Imitator", "拼接模式为空，兜底为「自动(跟随方案最优)」")

        # —— 一镜到底：生成模式即「零转场连续长镜头」，拼接必须走 seamless(去重硬切) ——
        # 否则 auto 会在段边界连续(差异<30)时误选交叉淡化，产生可见溶解转场，违背一镜到底本意。
        # 这里读取段落计划的生成模式，强制覆盖拼接模式（用户误选交叉淡化也无效）。
        try:
            _plan_mode = ""
            _plan_single_pass = False
            _plan_seamless = ""
            _plan_precision = "fp8"
            _plan_raw = 段落计划.strip()
            if _plan_raw:
                _pd = json.loads(_plan_raw)
                _plan_mode = _pd.get("mode", "")
                _plan_single_pass = bool(_pd.get("single_pass", False))
                # 统一『连贯方案』后，以 seamless_plan 为权威；旧 plan 缺该字段时
                # 由 continuity_strategy 反推（multi_seg→A / single_pass→C / warm_start→暖启动）。
                _plan_seamless = _pd.get("seamless_plan", "")
                if not _plan_seamless:
                    _c = _pd.get("continuity_strategy", "")
                    _plan_seamless = {
                        CONTINUITY_MULTI_SEG: SEAMLESS_PLAN_A,
                        CONTINUITY_SINGLE_PASS: SEAMLESS_PLAN_C,
                        CONTINUITY_WARM_START: CONTINUITY_WARM_START,
                    }.get(_c, SEAMLESS_PLAN_A)
                _plan_precision = _pd.get("model_precision", "fp8")
        except Exception:
            _plan_mode = ""
            _plan_single_pass = False
            _plan_seamless = ""
            _plan_precision = "fp8"
        # —— 自动(跟随方案最优)：显式解析，依据 后端/连贯方案/一镜到底 选最优拼接 ——
        # 用户在「分段规划」选完 A/B/暖启动后，本节点保持「自动」即可，无需二次选择：
        #   · SCAIL-2 路线 多段 → 帧锚定一镜到底(安全网)：跨段连续主路径已在生成侧
        #     由 transition_video 尾帧硬冻结续写硬保证（肥猴SQR同款，前段尾帧=后段
        #     起始）；接缝失配极小时拼接自动退化为纯顺序拼接，仅失配明显才淡化兜底。
        #   · 骨骼路线 多段（独立去噪、无生成侧连续）→ 交叉淡化像素平滑过渡。
        # 其余显式选择（帧锚定/无缝/交叉淡化/潜空间/硬切/ffmpeg转场）一律尊重，不覆盖。
        _continuity_capable = (生成后端 == "SCAIL-2 路线")
        if 拼接模式 == STITCH_AUTO:
            if (not _plan_single_pass) and _continuity_capable:
                info("Imitator", "自动：SCAIL-2 路线多段 → 帧锚定一镜到底(生成侧transition硬冻结为主·拼接淡化兜底)")
                拼接模式 = STITCH_FRAME_ANCHOR
            elif (not _plan_single_pass) and (not _continuity_capable):
                info("Imitator", "自动：骨骼路线多段(独立去噪) → 交叉淡化像素平滑过渡")
                拼接模式 = STITCH_CROSS_DISSOLVE
            # 单遍(_plan_single_pass) 无需拼接，下方直接成片，拼接模式留 auto 无害。

        # —— 一镜到底：拼接模式防呆（仅拦截真正破坏连贯的「硬切」）——
        # 一镜到底=连续长镜头；硬切会暴露段边界跳变，故强制回退 重叠混合。
        # ffmpeg转场(xfade 真·交叉溶解) 与 交叉淡化 本质都是「平滑过渡」，恰是一镜到底想要的丝滑，
        # 故放行（用户选 ffmpeg转场+淡入淡出 即恢复此前最喜欢的 0.5s 平滑接缝）。
        if _plan_mode == SEGMENT_MODE_ONE_SHOT and 拼接模式 == STITCH_HARD_CUT:
            info("Imitator", "生成模式=一镜到底，硬切会暴露段边界跳变，强制回退「无缝一镜到底(重叠混合)」")
            拼接模式 = STITCH_SEAMLESS_BLEND
        # —— 帧锚定尊重规则（2026-08-15 起拼接阶段确定性锚定已生效，据后端能力分流）——
        # 帧锚定(STITCH_FRAME_ANCHOR / STITCH_SEAMLESS) 在拼接阶段做「尾帧续接淡化」
        # （无重复帧 + 按接缝失配自适应软化窗），与后端无关、对任一路线都生效 →
        # 一律尊重用户显式选择（真无缝），不在防呆里降级。
        # 仅 STITCH_LATENT_BLEND（潜空间事后混合，4 步蒸馏下名不副实）→ 多段时升级为交叉淡化，
        # 避免无真实生成侧连续时裸混合跳变。骨骼路线选了 latent_blend 同样走此降级。
        if (not _plan_single_pass) and (not _continuity_capable) and 拼接模式 == STITCH_LATENT_BLEND:
            info("Imitator", "骨骼路线多段 + 潜空间混合名不副实，升级为「交叉淡化」平滑过渡")
            拼接模式 = STITCH_CROSS_DISSOLVE
        elif (not _plan_single_pass) and 拼接模式 == STITCH_LATENT_BLEND:
            # latent_blend 仅在段间真正重叠(overlap_prev>0, 生成侧连续)时才有意义；
            # 独立去噪段 overlap_prev=0 → 纯硬切。统一升级为交叉淡化，避免无重叠时裸硬切跳变。
            info("Imitator", "多段 latent_blend 升级为「交叉淡化」平滑过渡")
            拼接模式 = STITCH_CROSS_DISSOLVE
        # 帧锚定/无缝(STITCH_FRAME_ANCHOR / STITCH_SEAMLESS)：尊重用户选择，不覆盖（对任一后端真无缝）。

        # —— 单一效果模块，份传给两侧（本节点核心价值）——
        effects = 效果模块 or ""

        # 1) 生成相位：调用链式执行引擎，效果模块作用于每段 prompt/params
        runner = YunjiiSegmentRunner()
        执行日志 = ""  # 防御：run() 抛异常时 except 分支需引用，先置空避免 UnboundLocalError 掩盖真实报错
        try:
            执行结果, 执行日志, 完成状态 = runner.run(
                段落计划, 工作流模板, 执行模式, 最大重试,
                生成后端=生成后端, 视频路径=视频路径, 参考图=参考图, 姿态图=姿态图,
                人物参考图=人物参考图, 起始段=起始段, 效果模块=effects, ComfyUI地址=ComfyUI地址,
                连贯方案=_plan_seamless, 模型精度=_plan_precision,
            )
        except Exception as _exc:
            import traceback as _tb
            _detail = "".join(_tb.format_exception_only(type(_exc), _exc)).strip()
            _stack = _tb.format_exc()
            node_error("Imitator", "生成阶段异常: %s" % _detail)
            info("Imitator", "异常堆栈:\n%s", _stack)
            node_end("Imitator", "生成失败(异常)")
            return ("", f"{执行日志}\n[异常] {_detail}", False, _make_cover("")[1])

        # 仅规划：runner 已返回计划摘要，直接短路，不做拼接
        if 执行模式 == "仅规划":
            info("Imitator", "仅规划模式，跳过生成与拼接")
            node_end("Imitator", "仅规划")
            return ("", 执行日志, 完成状态, _make_cover("")[1])

        if not 完成状态:
            # 透传 runner 返回的真实错误信息（执行结果里含具体原因），不再笼统吞掉
            _err_msg = (执行结果 or "未知原因").strip()
            node_error("Imitator", "生成阶段失败: %s" % _err_msg)
            node_end("Imitator", "生成失败")
            return ("", f"{执行日志}\n[生成失败] {_err_msg}", False, _make_cover("")[1])

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
                    # 内联执行会改写外层 prompt 历史上下文，节点 return ui 落不进「已生成」；
                    # 故最终成片显式登记一条独立历史条目（与段视频同源通道）。
                    try:
                        register_video_to_history(_final)
                    except Exception as _e:
                        warn("Imitator", "方案C 最终成片前端历史补登失败(不影响出片): %s", _e)
                    _first, _cover = _make_cover(_final)
                    return {"ui": _build_output_ui(_final, _first),
                            "result": (_final, f"{执行日志}\n---\n✅ 一镜到底单次超长生成，单段即成片，无需拼接", True, _cover)}
            except Exception as e:
                warn("Imitator", "单次超长结果解析失败，回退拼接: %s", e)

        # 2) 拼接相位：调用无缝拼接，效果模块作用于成片（超分/插帧/调色等）
        #    ⚠️ stitch() 已重构为返回 dict：{"ui":..., "result":(视频路径, 报告, cover)}，
        #    不再返回裸 3 元组。直接解包 3 值会导致
        #    ValueError: not enough values to unpack (expected 3, got 2)。
        stitcher = YunjiiSegmentStitcher()
        _stitch_ret = stitcher.stitch(
            执行结果, 拼接模式, 淡化帧数, 输出文件名,
            音频源=音频源, 效果模块=effects,
            转场类型=转场类型, 转场时长=转场时长,
        )
        最终视频路径, 拼接报告, _stitcher_cover = _stitch_ret["result"]
        # stitch 内部已做前端历史登记并返回 ui；透传给节点 return，
        # 确保成片稳定进「已生成」（内联执行会改写外层 history）。
        _stitch_ui = _stitch_ret.get("ui")
        # 内联执行会改写外层 prompt 历史上下文，节点 return ui 落不进「已生成」；
        # 故最终成片显式登记一条独立历史条目（与段视频同源通道），确保稳定可见。
        try:
            register_video_to_history(最终视频路径)
        except Exception as _e:
            warn("Imitator", "最终成片前端历史补登失败(不影响出片): %s", _e)

        if not 最终视频路径:
            node_error("Imitator", "拼接阶段未产出视频")
            node_end("Imitator", "拼接失败")
            return ("", f"{执行日志}\n---\n{拼接报告}", False, _make_cover("")[1])

        # 合并两侧日志，便于一次查看全链路
        combined = f"{执行日志}\n---\n{拼接报告}" if 执行日志 else 拼接报告
        info("Imitator", "完美模仿完成: %s", 最终视频路径)
        node_end("Imitator", "完成")
        _first, _cover = _make_cover(最终视频路径)
        _ui = _stitch_ui or _build_output_ui(最终视频路径, _first)
        return {
            "ui": _ui,
            "result": (最终视频路径, combined, True, _cover),
        }
