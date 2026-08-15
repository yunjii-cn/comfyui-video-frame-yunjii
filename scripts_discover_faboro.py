import json, glob, os

WF_DIR = r"F:\ComfyUI_heihe\ComfyUI\user\default\workflows\FaboroHacks"
files = glob.glob(os.path.join(WF_DIR, "*.json"))
print("FILES:", [os.path.basename(f) for f in files])

for p in files:
    print("\n==========", os.path.basename(p), "==========")
    d = json.load(open(p, encoding="utf-8"))
    nodes = d.get("nodes", [])
    links = d.get("links", [])
    print("nodes=%d links=%d" % (len(nodes), len(links)))
    # build link map: link_id -> (origin_id, origin_slot)
    link_map = {}
    for l in links:
        # [id, origin_id, origin_slot, target_id, target_slot, type]
        link_map[l[0]] = (l[1], l[2])
    # inventory by type
    from collections import Counter
    cnt = Counter(n.get("type") for n in nodes)
    for t, c in cnt.most_common():
        print("  %3d x %s" % (c, t))
    # find loaders (Load*) and print their widget values
    print("--- LOADERS ---")
    for n in nodes:
        t = n.get("type") or ""
        if "Load" in t or "load" in t or "Checkpoint" in t or "VAE" in t or "CLIP" in t:
            wv = n.get("widgets_values")
            print("  [%s] id=%s title=%s" % (t, n.get("id"), n.get("title")))
            if wv is not None:
                print("      widgets_values:", json.dumps(wv, ensure_ascii=False)[:400])
    # find SCAIL2 node
    print("--- SCAIL2 GENERATOR NODE ---")
    for n in nodes:
        if "SCAIL2" in (n.get("type") or ""):
            print("  type=%s id=%s title=%s" % (n.get("type"), n.get("id"), n.get("title")))
            for inp in n.get("inputs", []):
                link = inp.get("link")
                if link is not None and link in link_map:
                    oid, oslot = link_map[link]
                    print("    input %-28s -> node %s slot %s" % (inp.get("name"), oid, oslot))
                else:
                    print("    input %-28s (widget: %s)" % (inp.get("name"), json.dumps(inp.get("widget", inp.get("value")), ensure_ascii=False)[:120]))
            print("    widgets_values:", json.dumps(n.get("widgets_values"), ensure_ascii=False)[:600])
