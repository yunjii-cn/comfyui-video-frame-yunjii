import json, time, urllib.request, os

PROMPT_ID = open("/tmp/faboro_prompt_id.txt").read().strip()
OUT_DIR = r"F:\ComfyUI_heihe\ComfyUI\output"
LOG = "/tmp/faboro_monitor.log"

def get(url):
    try:
        return json.loads(urllib.request.urlopen(url, timeout=10).read().decode())
    except urllib.error.HTTPError as e:
        return {"_http": e.code}
    except Exception as e:
        return {"_err": str(e)}

start = time.time()
deadline = start + 20*60  # 20 min max
last = ""
with open(LOG, "w") as f:
    f.write("monitor start %s\n" % time.strftime("%H:%M:%S"))
while time.time() < deadline:
    # progress: queue
    q = get("http://127.0.0.1:8188/queue")
    running = []
    if isinstance(q, dict):
        running = [r[1] for r in (q.get("queue_running") or [])]
    # history
    h = get("http://127.0.0.1:8188/history/" + PROMPT_ID)
    status = ""
    if isinstance(h, dict) and PROMPT_ID in h:
        entry = h[PROMPT_ID]
        if "outputs" in entry:
            outputs = entry["outputs"]
            # find video
            vids = []
            for nid, o in outputs.items():
                for v in (o.get("videos") or []):
                    vids.append(v.get("filename"))
            status = "DONE videos=%s" % vids
        elif "status" in entry:
            st = entry["status"]
            if st.get("status_str") == "error" or "messages" in st:
                # extract error
                msgs = st.get("messages", [])
                err = str(msgs)
                status = "ERROR " + err[:2000]
        else:
            status = "history-entry-no-outputs"
    else:
        status = "running (queue_running=%d)" % len(running)
    if status != last:
        line = "%s | %s\n" % (time.strftime("%H:%M:%S"), status)
        with open(LOG, "a") as f:
            f.write(line)
        print(line, end="")
        last = status
        if status.startswith("DONE") or status.startswith("ERROR"):
            break
    time.sleep(15)

# final: list output dir for our prefix
if status.startswith("DONE"):
    hits = []
    for root, _, files in os.walk(OUT_DIR):
        for fn in files:
            if fn.startswith("yunjii_faboro_test") or "faboro" in fn:
                hits.append(os.path.join(root, fn))
    with open(LOG, "a") as f:
        f.write("OUTPUT FILES:\n" + "\n".join(hits) + "\n")
    print("OUTPUT FILES:\n" + "\n".join(hits))
