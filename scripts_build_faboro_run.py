import json, glob, os, urllib.request, uuid

# ---- load object_info for validation ----
OI = {}
for f in glob.glob("/tmp/oi_*.json"):
    cls = os.path.basename(f)[3:-5]
    d = json.load(open(f))
    if cls in d:
        OI[cls] = d[cls]

def valid_inputs(class_type):
    info = OI.get(class_type, {})
    inp = info.get("input", {})
    names = set((inp.get("required", {}) or {}).keys())
    names |= set((inp.get("optional", {}) or {}).keys())
    return names

def check(class_type, inputs):
    ok = valid_inputs(class_type)
    for k in inputs:
        if k not in ok:
            raise SystemExit(f"INVALID input '{k}' for {class_type}. valid={sorted(ok)}")

# ---- Build minimal SCAIL2 prompt (reuses FaboroHacks wiring) ----
SCAIL_MODEL = "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"
LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
DRIVE_VIDEO = "poses.mp4"
REF_IMAGE = "example.png"

P = {}
def add(nid, ct, inputs):
    check(ct, inputs)
    P[str(nid)] = {"class_type": ct, "inputs": inputs}

add(1, "DiffusionModelLoaderKJ", {
    "model_name": SCAIL_MODEL, "weight_dtype": "default", "compute_dtype": "default",
    "patch_cublaslinear": False, "sage_attention": "auto", "enable_fp16_accumulation": True})
add(2, "LoraLoaderModelOnly", {
    "model": ["1", 0], "lora_name": "wan\\lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors", "strength_model": 1.0})
add(3, "ModelSamplingSD3", {"model": ["2", 0], "shift": 5})
add(4, "CLIPLoader", {"clip_name": "Wan\\umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"})
add(5, "VAELoader", {"vae_name": "WAN\\wan_2.1_vae.safetensors"})
add(6, "CLIPVisionLoader", {"clip_name": "clip_vision_h.safetensors"})
add(7, "CheckpointLoaderSimple", {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"})
add(8, "KSamplerSelect", {"sampler_name": "euler_ancestral"})
add(9, "BasicScheduler", {"model": ["3", 0], "scheduler": "beta", "steps": 4, "denoise": 1.0})
add(10, "CLIPTextEncode", {"text": "face", "clip": ["7", 1]})
add(11, "VHS_LoadVideo", {
    "video": DRIVE_VIDEO, "force_rate": 16, "custom_width": 0, "custom_height": 0,
    "frame_load_cap": 49, "skip_first_frames": 0, "select_every_nth": 1})
add(12, "ImageResizeKJv2", {
    "image": ["11", 0], "width": 736, "height": 736, "upscale_method": "nearest-exact",
    "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 2})
add(13, "LoadImage", {"image": REF_IMAGE})
add(14, "ImageResizeKJv2", {
    "image": ["13", 0], "width": 736, "height": 736, "upscale_method": "nearest-exact",
    "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 2})
add(16, "SCAIL2ScheduledLongVideoWithSAM", {
    "model": ["3", 0], "clip": ["4", 0], "vae": ["5", 0], "sampler": ["8", 0], "sigmas": ["9", 0],
    "clip_vision": ["6", 0], "pose_video": ["12", 0],
    "segment_plan": "# frames | reference | prompt | negative | boundary_overlap\n49 | 1 | a woman dancing | low quality | 5",
    "seed": 1, "cfg": 1.0, "mode": "replacement", "max_frames": 49, "max_chunk_frames": 49,
    "overlap_frames": 5, "reference_count": 1, "color_correction": True,
    "object_indices": "", "reference_object_indices": "", "sort_by": "left_to_right",
    "sam_detection_threshold": 0.5, "sam_max_objects": 2, "sam_detect_interval": 2,
    "cache_mode": "disk",
    "sam_model": ["7", 0], "sam_conditioning": ["10", 0], "reference_1": ["14", 0]})
add(17, "VHS_VideoCombine", {
    "images": ["16", 0], "frame_rate": 16, "loop_count": 0,
    "filename_prefix": "yunjii_faboro_test", "format": "video/h264-mp4",
    "pingpong": False, "save_output": True})

payload = {"prompt": P, "client_id": str(uuid.uuid4())}
with open("/tmp/faboro_prompt.json", "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)
print("Built prompt with %d nodes. Posting..." % len(P))

req = urllib.request.Request("http://127.0.0.1:8188/prompt",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=30).read().decode()
    print("POST /prompt response:", resp)
    pid = json.loads(resp).get("prompt_id")
    if pid:
        open("/tmp/faboro_prompt_id.txt", "w").write(pid)
        print("prompt_id saved:", pid)
except urllib.error.HTTPError as e:
    print("HTTPError", e.code, e.read().decode())
except Exception as e:
    print("Error:", e)
