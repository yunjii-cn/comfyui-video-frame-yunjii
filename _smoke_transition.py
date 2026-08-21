# -*- coding: utf-8 -*-
"""冒烟测试：多段拼接根治方案（transition_video 尾帧硬冻结续写，肥猴SQR同款）

T1: planner SCAIL 多段 → 首尾相接(重叠0/每段≤81帧/边界连续)
T2: animateplus _inject_transition_video → VHS 注入参数正确/接线正确/占用跳过/段0不注入
T3: stitcher 帧锚定 → 接缝失配<0.02 纯顺序拼接(B=0)；失配大 → 自适应淡化兜底
T6: sanitize COMBO 非法值 → 节点默认值（模板/节点包版本漂移兜底）
T7: 文件名类 COMBO 绕过 sanitize + _fix_model_names 接管 lora_N（二测惨案）
T8: 主输出VHS=吃Decode输出的成片节点，防选中骨骼预览（三测只出骨骼视频根因）
运行: python_embeded/python.exe _smoke_transition.py
"""
import sys
import os
import json

COMFY_ROOT = r"f:\ComfyUI_heihe\ComfyUI"
PLUGIN_ROOT = r"f:\ComfyUI_heihe\ComfyUI\custom_nodes\comfyui-video-frame-yunjii"
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, PLUGIN_ROOT)

import cv2
import numpy as np
import folder_paths

from engine.planner import YunjiiSegmentPlanner
from engine.adapters.animateplus import (
    AnimatePlusSCAILAdapter, AnimatePlusNodeMap,
    TRANSITION_FRAMES, AP_VHS_LOADVIDEO,
)
from engine.types import SegmentInfo
from engine.stitcher import YunjiiSegmentStitcher

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def make_video(path, frames):
    h, w = 64, 64
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


# ---------------------------------------------------------------- T1 planner
print("=" * 60)
print("T1: planner SCAIL 多段 → 重叠32基线（2026-08-20 回滚后）")
seg_info = "原始视频: 16fps, 总帧数: 200\n镜头1: 帧0-200\n"
planner = YunjiiSegmentPlanner()
plan_json, n_seg, _summary = planner.plan(
    分段信息=seg_info, 运动提示词="镜头1: 走路",
    连贯方案="短视频·多段拼接（≤15秒，独立生成+接缝淡化）",
    每段最大帧数=81, 重叠帧数=8, 目标分辨率="832x480", 目标帧率=16,
    自适应参数=True, 生成后端="SCAIL-2 路线",
)
pd = json.loads(plan_json)
segs = pd["segments"]
check("T1 200帧切成多段(≥2)", len(segs) >= 2, f"n={len(segs)}")
check("T1 每段≤81帧", all(s["target_frames"] <= 81 for s in segs),
      str([s["target_frames"] for s in segs]))
check("T1 多段无缝重叠=UI值8(3efa21f基线,不锁32)",
      all(s.get("overlap_prev", 0) == 8 for s in segs[1:]),
      str([s.get("overlap_prev") for s in segs]))
contig = all(segs[i]["end_frame"] - 8 == segs[i + 1]["start_frame"]
             for i in range(len(segs) - 1))
check("T1 边界按重叠8交叠 end[i]-8==start[i+1]", contig,
      str([(s["start_frame"], s["end_frame"]) for s in segs]))

# 智能分段(smart_split)模式 → 锁 32（猴子基线，与 3efa21f 一致）
plan_json2, _, _ = planner.plan(
    分段信息=seg_info, 运动提示词="镜头1: 走路",
    连贯方案="分段转场·重叠混合（每段独立，适合做转场效果）",
    每段最大帧数=81, 重叠帧数=8, 目标分辨率="832x480", 目标帧率=16,
    自适应参数=True, 生成后端="SCAIL-2 路线",
)
segs2 = json.loads(plan_json2)["segments"]
check("T1 智能分段锁重叠32", len(segs2) >= 2 and all(
    s.get("overlap_prev", 0) == 32 for s in segs2[1:]),
    str([s.get("overlap_prev") for s in segs2]))

# ------------------------------------------------- T2 transition_video 注入
print("=" * 60)
print("T2: animateplus _inject_transition_video")
tmp = folder_paths.get_temp_directory()
os.makedirs(tmp, exist_ok=True)

# 上段成片: 40帧灰度渐变
prev_path = os.path.join(tmp, "smoke_prev_seg.mp4")
make_video(prev_path, [np.full((64, 64, 3), v, np.uint8) for v in range(0, 80, 2)])

adapter = AnimatePlusSCAILAdapter(folder_paths.get_output_directory())
node_map = AnimatePlusNodeMap()
node_map.animate_embeds = "10"

wf = {
    "10": {"class_type": "WanAnimatePlus SCAIL_2 Embeds",
           "inputs": {"num_frames": 81}},
    "20": {"class_type": "WanAnimatePlus Sampler", "inputs": {}},
}
seg1 = SegmentInfo(index=1, start_frame=81, end_frame=162, target_frames=81)

ok = adapter._inject_transition_video(wf, node_map, prev_path, seg1)
check("T2 注入成功返回True", ok)

vhs_nodes = [k for k, v in wf.items() if isinstance(v, dict)
             and v.get("class_type") == AP_VHS_LOADVIDEO]
check("T2 新增VHS_LoadVideo节点", len(vhs_nodes) == 1, str(vhs_nodes))
if vhs_nodes:
    vi = wf[vhs_nodes[0]]["inputs"]
    total = adapter._video_frame_count(prev_path)
    n_expect = min(TRANSITION_FRAMES, total)
    check("T2 frame_load_cap=尾21帧", vi.get("frame_load_cap") == n_expect,
          f"cap={vi.get('frame_load_cap')}, expect={n_expect}")
    check("T2 skip_first_frames=total-21",
          vi.get("skip_first_frames") == max(0, total - n_expect),
          f"skip={vi.get('skip_first_frames')}, expect={max(0, total - n_expect)}")
    check("T2 embeds.transition_video已接线",
          wf["10"]["inputs"].get("transition_video") == [vhs_nodes[0], 0],
          str(wf["10"]["inputs"].get("transition_video")))

# 段0 不注入
wf2 = {"10": {"class_type": "WanAnimatePlus SCAIL_2 Embeds", "inputs": {}}}
seg0 = SegmentInfo(index=0, start_frame=0, end_frame=81, target_frames=81)
ok0 = adapter._inject_transition_video(wf2, node_map, prev_path, seg0)
check("T2 段0不注入(返回False)", ok0 is False)
check("T2 段0未新增VHS节点",
      not any(isinstance(v, dict) and v.get("class_type") == AP_VHS_LOADVIDEO
              for v in wf2.values()))

# 模板已占用 → 跳过
wf3 = {"10": {"class_type": "WanAnimatePlus SCAIL_2 Embeds",
              "inputs": {"transition_video": ["99", 0]}}}
ok3 = adapter._inject_transition_video(wf3, node_map, prev_path, seg1)
check("T2 模板占用transition_video→跳过(False)", ok3 is False)

# --------------------------------------------------- T3 stitcher 帧锚定
print("=" * 60)
print("T3: stitcher 帧锚定（安全网语义）")
stitcher = YunjiiSegmentStitcher()


def run_anchor(v0_frames, v1_frames):
    p0 = make_video(os.path.join(tmp, "smoke_s0.mp4"), v0_frames)
    p1 = make_video(os.path.join(tmp, "smoke_s1.mp4"), v1_frames)
    report = []
    # monkeypatch _write_frames：挂实例上（无 self 绑定），内存计数，
    # 避免沙箱限制 output 目录读写
    captured = {}

    def fake_write(frames, output_path, fps, width, height):
        captured["n"] = len(frames)
        return output_path

    orig_write = stitcher._write_frames
    stitcher._write_frames = fake_write
    try:
        stitcher._stitch_videos_frame_anchor(
            [{"path": p0, "overlap_prev": 0, "segment_index": 0},
             {"path": p1, "overlap_prev": 0, "segment_index": 1}],
            "smoke_anchor", report, run_id="smoke_transition",
            blend_frames=8)
    finally:
        stitcher._write_frames = orig_write
    return captured.get("n", 0), report


# 场景A: 接缝连续(失配<0.02) → 纯顺序拼接
grad = [np.full((64, 64, 3), v, np.uint8) for v in range(0, 160, 4)]  # 40帧渐变
n_a, rep_a = run_anchor(grad[:20], grad[20:])
check("T3A 连续接缝总帧数=40(无丢帧)", n_a == 40, f"n={n_a}")
check("T3A 纯顺序拼接(无淡化)",
      any("纯顺序拼接" in r for r in rep_a), "; ".join(rep_a))

# 场景B: 接缝失配大(黑→白) → 自适应淡化兜底
n_b, rep_b = run_anchor(
    [np.zeros((64, 64, 3), np.uint8) for _ in range(20)],
    [np.full((64, 64, 3), 255, np.uint8) for _ in range(20)])
check("T3B 失配大总帧数=40(无丢帧)", n_b == 40, f"n={n_b}")
check("T3B 自适应淡化兜底生效",
      any("尾帧续接淡化" in r for r in rep_b), "; ".join(rep_b))

# --------------------------------- T4 runner 方案矩阵路由（显式方案，无隐式切换）
print("=" * 60)
print("T4: runner 方案矩阵路由（A不隐式切原生 / NATIVE显式切 / SQR/WARM不切）")
from engine import runner as runner_mod
import engine.adapters.scail2_native as scail2_native_mod

runner = runner_mod.YunjiiSegmentRunner()
calls = {"native": 0}


def fake_native(plan, *a, **kw):
    calls["native"] += 1
    return ("{}", "mock-native-ok", True)


runner._run_native_scail2 = fake_native
orig_avail = scail2_native_mod.is_native_scail2_available

LABEL_A = "短视频·多段拼接（≤15秒，独立生成+接缝淡化）"
LABEL_NATIVE = "长视频·原生调度无劣化（FaboroHacks同款：节点内多块锚定，一次成片）"
LABEL_SQR = "多段队列·硬冻结接段（肥猴同款：外部分段+上段尾帧锚定续写）"
LABEL_WARM = "暖启动·潜空间续写（Tier2：WanAnimatePlus上段末帧喂回）"

# 多段拼接(A) + 原生包可用 → 不再隐式切原生（方案边界清晰化后的核心断言）
scail2_native_mod.is_native_scail2_available = lambda: True
try:
    try:
        runner.run(plan_json, "", "执行", 1,
                   生成后端="SCAIL-2 路线", 连贯方案=LABEL_A)
    except Exception:
        pass
finally:
    scail2_native_mod.is_native_scail2_available = orig_avail
check("T4 方案A不再隐式切原生", calls["native"] == 0, f"calls={calls['native']}")

# 原生调度方案 → 显式切 _run_native_scail2
calls["native"] = 0
try:
    res = runner.run(plan_json, "", "执行", 1,
                     生成后端="SCAIL-2 路线", 连贯方案=LABEL_NATIVE)
finally:
    pass
check("T4 原生调度方案→显式切原生", calls["native"] == 1,
      f"calls={calls['native']}")
check("T4 原生调度返回透传", res[1] == "mock-native-ok", str(res[1])[:40])

# 多段队列(SQR) → 不切原生（走 WanAnimatePlus 模板 + transition_video 注入）
calls["native"] = 0
try:
    try:
        runner.run(plan_json, "", "执行", 1,
                   生成后端="SCAIL-2 路线", 连贯方案=LABEL_SQR)
    except Exception:
        pass
finally:
    pass
check("T4 多段队列(SQR)不切原生", calls["native"] == 0, f"calls={calls['native']}")

# 暖启动策略 → 不切原生（保持 WanAnimatePlus 模板路线）
calls["native"] = 0
scail2_native_mod.is_native_scail2_available = lambda: True
try:
    try:
        runner.run(plan_json, "", "执行", 1,
                   生成后端="SCAIL-2 路线", 连贯方案=LABEL_WARM)
    except Exception:
        pass
finally:
    scail2_native_mod.is_native_scail2_available = orig_avail
check("T4 暖启动策略不切原生", calls["native"] == 0, f"calls={calls['native']}")

# --------------------------------- T5 planner SQR 段间重叠锁0（首尾相接）
print("=" * 60)
print("T5: planner 多段队列(SQR) 段间重叠=0 + 方案解析")
from engine.types import resolve_unified_plan, CONTINUITY_SQR, CONTINUITY_NATIVE
st, sp, _md = resolve_unified_plan(LABEL_SQR)
check("T5 SQR标签→strategy=sqr_queue", st == CONTINUITY_SQR, f"st={st}")
check("T5 SQR标签→seamless_plan=seamless_sqr", sp == "seamless_sqr", f"sp={sp}")
st2, sp2, _md2 = resolve_unified_plan(LABEL_NATIVE)
check("T5 NATIVE标签→strategy=native_scheduled", st2 == CONTINUITY_NATIVE, f"st={st2}")
check("T5 NATIVE标签→seamless_plan=seamless_native", sp2 == "seamless_native", f"sp={sp2}")

# planner SQR 段间重叠=0：直接跑 plan()，检查各段 overlap_prev
from engine.planner import YunjiiSegmentPlanner
_pl = YunjiiSegmentPlanner()
_seg_info = "原始视频: 16fps, 总帧数: 200\n镜头1: 帧0-200\n"
_pj, _nf, _log = _pl.plan(_seg_info, "测试", LABEL_SQR, 81, 8,
                          "832x480", 16, True,
                          生成后端="SCAIL-2 路线")
try:
    _plan = json.loads(_pj)
    _ovs = [s["overlap_prev"] for s in _plan["segments"]]
    check("T5 SQR planner段间重叠全0(首尾相接)",
          all(o == 0 for o in _ovs), f"ovs={_ovs}")
    _st = _plan.get("continuity_strategy", "")
    check("T5 SQR plan含sqr_queue策略", _st == "sqr_queue", f"st={_st}")
except Exception as e:
    check("T5 SQR planner段间重叠全0(首尾相接)", False, f"异常: {e}")

# --------------------------------- T6 COMBO 版本漂移兜底（SQR 首测崩溃根因）
print("=" * 60)
print("T6: sanitize COMBO 非法值 → 节点默认值（模板/节点包版本漂移兜底）")
from engine.adapters.scail import SCAILAdapter

# 结构对齐真实 /object_info：COMBO 声明首元素=选项列表，类型声明首元素=类型字符串
_fake_oi = {
    "BodyRatioMapperProportionTransfer": {
        "input": {
            "required": {"pose_keypoint": ["POSE_KEYPOINT"]},
            "optional": {
                "anchor_output_mode": (
                    ["single_frame_multi_person", "multi_frame_single_person"],
                    {"default": "single_frame_multi_person"}),
                "confidence_threshold": ("FLOAT", {"default": 0.30}),
            },
        },
    },
    "KSampler": {
        "input": {
            "required": {
                "sampler_name": (
                    ["euler", "euler_ancestral", "uni_pc"],
                    {"default": "euler"}),
                "steps": ("INT", {"default": 20}),
            },
        },
    },
}
_orig_oi = SCAILAdapter._OI_CACHE
SCAILAdapter._OI_CACHE = _fake_oi
try:
    api = {
        "1082": {"class_type": "BodyRatioMapperProportionTransfer",
                 "inputs": {"anchor_output_mode": False,       # 版本漂移错位值
                            "confidence_threshold": 0.3}},
        "10": {"class_type": "KSampler",
               "inputs": {"sampler_name": "euler", "steps": 20}},   # 合法，不动
    }
    out = SCAILAdapter._sanitize_combo_inputs(api)
    check("T6 非法COMBO重置为默认",
          out["1082"]["inputs"]["anchor_output_mode"] == "single_frame_multi_person",
          str(out["1082"]["inputs"]["anchor_output_mode"]))
    check("T6 合法COMBO不动",
          out["10"]["inputs"]["sampler_name"] == "euler",
          str(out["10"]["inputs"]["sampler_name"]))
    check("T6 FLOAT输入不受影响",
          out["1082"]["inputs"]["confidence_threshold"] == 0.3,
          str(out["1082"]["inputs"]["confidence_threshold"]))
finally:
    SCAILAdapter._OI_CACHE = _orig_oi

# --------------------------------- T7 文件名类 COMBO 不被 sanitize 重置 + lora_N 模糊匹配
# （2026-08-20 SQR 二测惨案：sanitize 把模板 VAE 重置成 FLUX VAE、文本编码器
#  重置成 Qwen-VL、LoRA 全置 none → 输出纯骨骼视频。文件名必须交给模糊匹配。）
print("=" * 60)
print("T7: 文件名类COMBO绕过sanitize + _fix_model_names接管lora_N")
_fake_oi7 = {
    "WanAnimatePlus VAELoader": {
        "input": {"required": {
            "model_name": (["FLUX.1\\UltraFlux-v1.safetensors",
                            "WAN\\Wan2_1_VAE_bf16.safetensors"],),
        }},
    },
    "WanAnimatePlus LoraSelectMulti": {
        "input": {"required": {
            "lora_1": (["none",
                        "wan\\lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
                        "wan\\Wan2.1 - Fun-14B-InP-HPS2.1.safetensors"],),
            "strength_1": ("FLOAT", {"default": 1.0}),
            "lora_4": (["none",
                        "wan\\Wan2.2 - I2V -Slop Bounce-Low-i2v-(弹跳lora不变脸).safetensors"],),
            "strength_4": ("FLOAT", {"default": 1.0}),
        }},
    },
}
_orig_oi7 = SCAILAdapter._OI_CACHE
SCAILAdapter._OI_CACHE = _fake_oi7
try:
    api = {
        "1242": {"class_type": "WanAnimatePlus VAELoader",
                 "inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors"}},  # 缺子目录前缀
        "1240": {"class_type": "WanAnimatePlus LoraSelectMulti",
                 "inputs": {
                     "lora_1": "Wan-Lighting\\lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                     "strength_1": 1.0,
                     "lora_4": "zzz_totally_unknown_lora.safetensors",  # 无近似 → none
                     "strength_4": 1.0}},
    }
    # ① sanitize 不得动文件名类 COMBO（哪怕值不在选项列表）
    out = SCAILAdapter._sanitize_combo_inputs(api)
    check("T7 sanitize不碰VAE文件名",
          out["1242"]["inputs"]["model_name"] == "Wan2_1_VAE_bf16.safetensors",
          str(out["1242"]["inputs"]["model_name"]))
    check("T7 sanitize不碰LoRA文件名",
          out["1240"]["inputs"]["lora_1"].endswith("rank128_bf16.safetensors"),
          str(out["1240"]["inputs"]["lora_1"]))
    # ② _fix_model_names 接管：VAE 模糊匹配、lora_1 模糊匹配、lora_4 回退 none
    SCAILAdapter._fix_model_names(out)
    check("T7 VAE模糊匹配补前缀",
          out["1242"]["inputs"]["model_name"] == "WAN\\Wan2_1_VAE_bf16.safetensors",
          str(out["1242"]["inputs"]["model_name"]))
    check("T7 lora_1模糊匹配distill",
          out["1240"]["inputs"]["lora_1"] ==
          "wan\\lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
          str(out["1240"]["inputs"]["lora_1"]))
    check("T7 lora_4无近似回退none",
          out["1240"]["inputs"]["lora_4"] == "none",
          str(out["1240"]["inputs"]["lora_4"]))
    check("T7 strength不受影响",
          out["1240"]["inputs"]["strength_1"] == 1.0,
          str(out["1240"]["inputs"]["strength_1"]))
finally:
    SCAILAdapter._OI_CACHE = _orig_oi7

# --------------------------------- T8 主输出VHS选真成片（SQR 三测惨案根因）
# （2026-08-20 SQR 三测：模板 3 个骨骼/姿态预览 VHS 均为 save_output=True 且前缀
#  'AnimateDiff' 不含姿态关键词，真成片 VHS(吃Decode输出)却 save_output=False →
#  旧「前缀+save_output」规则选中骨骼预览 → 内联只执行骨骼渲染链，采样器/Decode
#  根本没跑，成品=纯骨骼视频。修复：优先选 images 数据流来自 Decode 的 VHS。）
print("=" * 60)
print("T8: 主输出VHS=吃Decode输出的成片节点(防选中骨骼预览)")
_adapter8 = AnimatePlusSCAILAdapter(folder_paths.get_output_directory())


def _mk_api8(vhs312_images_src):
    """复刻肥猴模板 Set/Get 重连后的 API 拓扑（骨骼预览VHS + 真成片VHS + Decode链）"""
    return {
        "312": {"class_type": "VHS_VideoCombine",
                "inputs": {"images": vhs312_images_src, "audio": ["63", 1],
                           "frame_rate": ["500", 0],
                           "filename_prefix": "%date:yyyy-MM-dd%/x_Wanimate",
                           "save_output": False}},
        "1067": {"class_type": "VHS_VideoCombine",
                 "inputs": {"images": ["1087", 0], "frame_rate": ["500", 0],
                            "filename_prefix": "AnimateDiff", "save_output": True}},
        "1071": {"class_type": "VHS_VideoCombine",
                 "inputs": {"images": ["1092", 0], "frame_rate": ["500", 0],
                            "filename_prefix": "AnimateDiff", "save_output": True}},
        "1087": {"class_type": "BodyRatioMapperSDPoseRender",
                 "inputs": {"pose_keypoint": ["1082", 0]}},
        "1082": {"class_type": "BodyRatioMapperProportionTransfer",
                 "inputs": {"pose_keypoint": ["1092", 0]}},
        "1092": {"class_type": "PoseAndFaceDetection",
                 "inputs": {"video": ["63", 0]}},
        "1262": {"class_type": "WanAnimatePlus Decode",
                 "inputs": {"samples": ["1260", 0], "vae": ["1242", 0]}},
        "1260": {"class_type": "WanAnimatePlus SamplerFromSettings",
                 "inputs": {"embeds": ["1263", 0], "model": ["1238", 0]}},
        "1263": {"class_type": "WanAnimatePlus AnimateEmbeds",
                 "inputs": {"pose_images": ["1092", 0], "ref_images": ["651", 0]}},
        "1238": {"class_type": "WanAnimatePlus ModelLoader", "inputs": {}},
        "1242": {"class_type": "WanAnimatePlus VAELoader", "inputs": {}},
        "63": {"class_type": "VHS_LoadVideo", "inputs": {}},
        "651": {"class_type": "LoadImage", "inputs": {}},
        "500": {"class_type": "VHS_VideoInfoLoaded", "inputs": {}},
    }


# ① 直连：真成片 VHS.images 直接接 Decode
nm8 = _adapter8.discover_nodes(_mk_api8(["1262", 0]))
check("T8 直连Decode→选真成片VHS(312)",
      nm8.video_combine == "312", f"got={nm8.video_combine}")
check("T8 旧规则会误选骨骼预览(回归对照)",
      _adapter8._select_primary_vhs(
          [("312", "%date:yyyy-MM-dd%/x_Wanimate", False),
           ("1067", "AnimateDiff", True),
           ("1071", "AnimateDiff", True)]) == "1067",
      "prefix规则应选1067(证明T8必要性)")

# ② 间接：真成片 VHS.images 经 ImageResize 中转接 Decode
api8b = _mk_api8(["900", 0])
api8b["900"] = {"class_type": "ImageResizeKJv2", "inputs": {"image": ["1262", 0]}}
nm8b = _adapter8.discover_nodes(api8b)
check("T8 间接(经Resize)上溯Decode→选真成片VHS",
      nm8b.video_combine == "312", f"got={nm8b.video_combine}")

# ③ 回退：无 Decode 数据流 → 维持旧前缀规则（1067）
api8c = _mk_api8(["1087", 0])  # 真成片VHS也吃骨骼渲染（异常拓扑）
api8c.pop("1262"); api8c.pop("1260")
nm8c = _adapter8.discover_nodes(api8c)
check("T8 无Decode数据流→回退旧规则(1067)",
      nm8c.video_combine == "1067", f"got={nm8c.video_combine}")

# ④ 真实模板全链路：prepare(手术+Set/Get重连) + discover → 必须选中 312
try:
    _tpl = json.load(open(os.path.join(PLUGIN_ROOT, "workflows",
                                       "Tier2_WanAnimatePlus_Animate_template.json"),
                          encoding="utf-8"))
    _wf8 = _adapter8.prepare_workflow(_tpl)
    _nm8 = _adapter8.discover_nodes(_wf8)
    check("T8 真模板discover选312(真成片,吃Decode)",
          _nm8.video_combine == "312", f"got={_nm8.video_combine}")
    check("T8 真模板driving/ref不回退",
          _nm8.driving_video == "63" and _nm8.ref_image == "651",
          f"dv={_nm8.driving_video}, ref={_nm8.ref_image}")
except Exception as e:
    check("T8 真模板discover选312(真成片,吃Decode)", False,
          f"环境异常: {e}")

# ------------------------------------------------------------------ 汇总
print("=" * 60)
n_pass = sum(RESULTS)
print(f"RESULT: {n_pass}/{len(RESULTS)} passed")
sys.exit(0 if n_pass == len(RESULTS) else 1)
