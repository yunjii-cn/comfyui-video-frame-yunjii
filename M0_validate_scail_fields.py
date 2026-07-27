#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M0 字段校验脚本（在已装 SCAIL-2 的 ComfyUI 机器上运行）。

作用：
  连接本机 ComfyUI，拉取 WanSCAILToVideo / SCAIL2ColoredMask / SAM3_VideoTrack 的
  真实 INPUT_TYPES，与我们 engine/adapters/scail.py 里的字段假设逐项比对，
  输出「字段是否存在 / 真实名是什么」，把 M0 的「字段回填」从猜变成精确对齐。

用法：
  python M0_validate_scail_fields.py --url http://127.0.0.1:8188
  （ComfyUI 默认 8188；若改过端口按实际填）

增强（本版）：
  --save-json FILE   把真实节点的 INPUT_TYPES 导出到文件（默认 m0_scail_fields_dump.json）。
                     即使出现 MISS，也只需把该文件发回，即可精确回填 SCAIL_FIELD_MAP，无需手动复制。
  自动发现            若假定节点名未命中，会自动列出所有含 SCAIL/SAM 的节点并一并导出，避免改名漏判。

依赖：仅标准库（urllib / json / argparse / datetime），无需额外安装。
"""

import argparse
import json
import sys
import datetime
import urllib.request

# 我们 scail.py 里的字段假设（优先从适配器 import，失败则用内嵌副本，保证一定能跑）
EMBEDDED = {
    "WanSCAILToVideo": [
        "pose_video", "reference_image", "previous_frames",
        "previous_frame_count", "segment_index", "frame_count",
        "prompt", "replace_mode", "width", "height",
    ],
    "SCAIL2ColoredMask": [],
    "SAM3_VideoTrack": [],
}


def load_assumed():
    """优先从 engine.adapters.scail 读取 SCAIL_FIELD_MAP；失败则用内嵌副本。"""
    try:
        sys.path.insert(0, ".")
        from engine.adapters.scail import (
            SCAIL_FIELD_MAP, SCAIL_CORE, SCAIL_MASK, SCAIL_SAM3_TRACK,
        )
        nodes = {
            SCAIL_CORE: list(SCAIL_FIELD_MAP.values()),
            SCAIL_MASK: [],
            SCAIL_SAM3_TRACK: [],
        }
        print("[info] 已从 engine.adapters.scail 读取字段假设")
        return nodes
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 无法 import scail 适配器({e})，改用内嵌字段假设")
        return {k: list(v) for k, v in EMBEDDED.items()}


def fetch_object_info(base):
    url = base.rstrip("/") + "/object_info"
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "yunjii-m0-validate"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect_inputs(node_info):
    """返回 (required_dict, optional_dict)，与 ComfyUI INPUT_TYPES 结构对齐。"""
    inp = (node_info or {}).get("input", {}) or {}
    req = inp.get("required", {}) or {}
    opt = inp.get("optional", {}) or {}
    return req, opt


def _typeinfo(v):
    """把 INPUT_TYPES 的字段值压缩成可读类型名。"""
    if isinstance(v, (list, tuple)) and v:
        t = v[0]
        if isinstance(t, list):  # COMBO 选项列表
            return "COMBO"
        return str(t)
    return str(v)


def dump_targets(obj, assumed, base, path):
    """导出假定节点 + 所有 SCAIL/SAM 候选节点的真实 INPUT_TYPES 到 JSON。"""
    dump = {
        "base_url": base,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "nodes": {},
    }
    candidates = [n for n in obj if "SCAIL" in n.upper() or "SAM" in n.upper()]
    names_to_dump = set(candidates)
    for node in assumed:
        if node in obj:
            names_to_dump.add(node)

    for name in sorted(names_to_dump):
        ni = obj.get(name)
        req, opt = collect_inputs(ni)
        dump["nodes"][name] = {
            "found": True,
            "inputs": {
                "required": {k: _typeinfo(v) for k, v in req.items()},
                "optional": {k: _typeinfo(v) for k, v in opt.items()},
            },
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path, candidates


def main():
    ap = argparse.ArgumentParser(description="SCAIL-2 字段校验（M0 准备）")
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI 地址")
    ap.add_argument(
        "--save-json", default="m0_scail_fields_dump.json",
        help="把真实节点 INPUT_TYPES 导出到此文件（发回即可回填，无需手动复制字段）",
    )
    args = ap.parse_args()

    assumed = load_assumed()
    try:
        info = fetch_object_info(args.url)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 无法连接 ComfyUI ({args.url}): {e}")
        print("        请确认 ComfyUI 已启动、SCAIL-2 节点已安装，且 --url 端口正确。")
        print("        若 ComfyUI 在远程/容器，请把 --url 指向其可访问地址。")
        sys.exit(2)

    obj = info.get("object_info", info)
    print(f"\n=== 已安装节点总数: {len(obj)} ===\n")

    all_ok = True
    for node, fields in assumed.items():
        print(f"--- {node} ---")
        real = obj.get(node)
        if not real:
            print(f"  [MISS] 节点 {node} 未在 ComfyUI 中找到！")
            print(f"         可能 class_type 改名或节点未安装。下方列出含 SCAIL/SAM 的节点帮助定位。")
            all_ok = False
            hits = [n for n in obj if "SCAIL" in n.upper() or "SAM" in n.upper()]
            if hits:
                print(f"         疑似节点: {hits}")
            continue

        real_inputs = {}
        req, opt = collect_inputs(real)
        real_inputs.update(req)
        real_inputs.update(opt)
        real_field_names = set(real_inputs.keys())

        for f in sorted(fields):
            if f in real_field_names:
                print(f"  [OK]   {f}")
            else:
                print(f"  [MISS] {f}  -> 真实节点无此字段（见下方真实清单）")
                all_ok = False

        print(f"  真实输入字段清单({len(real_field_names)}):")
        for f in sorted(real_field_names):
            print(f"    - {f}")

    # 导出真实 INPUT_TYPES 到 JSON（发回即可精确回填）
    try:
        path, cands = dump_targets(obj, assumed, args.url, args.save_json)
        print(f"\n[OK] 已导出真实节点 INPUT_TYPES -> {path}")
        print(f"     把该文件发回给我，即可精确回填 SCAIL_FIELD_MAP（无需手动复制字段）。")
        if cands:
            print(f"     已包含 {len(cands)} 个 SCAIL/SAM 候选节点: {cands}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 导出 JSON 失败: {e}")

    print("\n=== 校验结果 ===")
    if all_ok:
        print("[PASS] 所有假设字段均与真实节点匹配，SCAIL_FIELD_MAP 可直接使用。")
        print("       下一步：runner 选「生成后端=SCAIL-2 路线」端到端跑首段验证。")
    else:
        print("[ACTION] 存在 MISS 项：把上面生成的 m0_scail_fields_dump.json 发回，")
        print("         我据此对齐 scail.py 的 SCAIL_FIELD_MAP 后，再做端到端验证。")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
