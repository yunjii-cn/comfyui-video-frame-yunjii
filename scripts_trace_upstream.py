import json, os
from collections import deque

WF = r"F:\ComfyUI_heihe\ComfyUI\user\default\workflows\FaboroHacks\ComfyUI+SCAIL+2+Face+Detailer+视频换脸+公开版.json"
d = json.load(open(WF, encoding="utf-8"))
nodes = {n["id"]: n for n in d["nodes"]}
# link_id -> (origin_id, origin_slot)
link_map = {l[0]: (l[1], l[2]) for l in d["links"]}

def upstream(start_ids):
    seen = set()
    q = deque(start_ids)
    while q:
        nid = q.popleft()
        if nid in seen: continue
        seen.add(nid)
        n = nodes.get(nid)
        if not n: continue
        for inp in n.get("inputs", []):
            lk = inp.get("link")
            if lk is not None and lk in link_map:
                q.append(link_map[lk][0])
    return seen

target = 92  # SCAIL2ScheduledLongVideoWithSAM in 视频换脸
sub = upstream([target])
print("subgraph node count:", len(sub))
# show SCAIL2 node's direct inputs with origins
print("\n=== SCAIL2 node 92 direct inputs ===")
for inp in nodes[92].get("inputs", []):
    lk = inp.get("link")
    if lk is not None and lk in link_map:
        oid, oslot = link_map[lk]
        on = nodes.get(oid, {})
        print("  %-26s -> node %s (%s) slot %s" % (inp["name"], oid, on.get("type"), oslot))
    else:
        print("  %-26s (widget)" % inp["name"])

# Dump every node in subgraph: type + widgets_values (truncated)
print("\n=== subgraph node details ===")
for nid in sorted(sub):
    n = nodes[nid]
    wv = n.get("widgets_values")
    wvs = json.dumps(wv, ensure_ascii=False)
    if len(wvs) > 300: wvs = wvs[:300] + "..."
    print("[%s] %s  wv=%s" % (nid, n.get("type"), wvs))
