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

输出示例：
  [OK]   WanSCAILToVideo.pose_video         -> 存在
  [MISS] WanSCAILToVideo.driving_video      -> 真实节点无此字段
  ...
  末尾打印每个节点的完整输入字段清单，方便直接复制回填到 scail.py 的 SCAIL_FIELD_MAP。

依赖：仅标准库（urllib / json / argparse），无需额外安装。
"""

import argparse
import json
import sys
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
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="SCAIL-2 字段校验（M0 准备）")
    ap.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI 地址")
    args = ap.parse_args()

    assumed = load_assumed()
    try:
        info = fetch_object_info(args.url)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 无法连接 ComfyUI ({args.url}): {e}")
        print("        请确认 ComfyUI 已启动、SCAIL-2 节点已安装，且 --url 端口正确。")
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
        for section in ("required", "optional"):
            real_inputs.update((real.get("input", {}) or {}).get(section, {}) or {})
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

    print("\n=== 校验结果 ===")
    if all_ok:
        print("[PASS] 所有假设字段均与真实节点匹配，SCAIL_FIELD_MAP 可直接使用。")
    else:
        print("[ACTION] 存在 MISS 项：把上面真实字段清单与 scail.py 的 SCAIL_FIELD_MAP 对齐后，")
        print("         再运行 runner（SCAIL-2 路线）做端到端验证。")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
