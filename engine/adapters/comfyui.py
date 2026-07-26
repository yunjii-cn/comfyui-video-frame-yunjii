import json
import os
import urllib.request
import urllib.error
import time
import logging

from ..types import NodeMap
from ..debug_log import debug, info, warn, error

logger = logging.getLogger(__name__)


class GenerationAdapter:
    def submit(self, task):
        raise NotImplementedError

    def wait(self, task_id):
        raise NotImplementedError

    def extract_frame(self, result, frame_idx):
        raise NotImplementedError

    def discover_nodes(self, workflow):
        raise NotImplementedError


class ComfyUIAdapter(GenerationAdapter):
    def __init__(self, host="127.0.0.1:8188", client_id="yunjii_v2v"):
        self.host = host
        self.client_id = client_id
        self._base_url = f"http://{host}"

    def submit(self, workflow_dict):
        payload = json.dumps({
            "prompt": workflow_dict,
            "client_id": self.client_id,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                prompt_id = data.get("prompt_id", "")
                logger.info("Submitted prompt: %s", prompt_id)
                return prompt_id
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error("Submit failed HTTP %d: %s", e.code, body)
            raise RuntimeError(f"Queue API submit failed (HTTP {e.code}): {body}") from e
        except urllib.error.URLError as e:
            logger.error("Submit connection failed: %s", e.reason)
            raise RuntimeError(f"Cannot connect to ComfyUI at {self._base_url}: {e.reason}") from e

    def wait(self, prompt_id, poll_interval=2.0, timeout=1800):
        start = time.time()
        last_log_time = start
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                logger.error("Prompt %s timed out after %ds", prompt_id, timeout)
                error("ComfyUIAdapter", "Prompt %s 超时 (%ds/%ds)", prompt_id, int(elapsed), timeout)
                return {"status": "timeout", "prompt_id": prompt_id}

            now = time.time()
            if now - last_log_time >= 15:
                info("ComfyUIAdapter", "等待prompt %s... 已等待%ds/%ds", prompt_id, int(elapsed), timeout)
                last_log_time = now

            try:
                url = f"{self._base_url}/history/{prompt_id}"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    history = json.loads(resp.read().decode("utf-8"))
            except Exception:
                time.sleep(poll_interval)
                continue

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                status_str = status.get("status_str", "")

                if status.get("completed", False):
                    logger.info("Prompt %s completed", prompt_id)
                    info("ComfyUIAdapter", "Prompt %s 完成! 耗时%ds", prompt_id, int(elapsed))
                    return {"status": "success", "prompt_id": prompt_id, "history": entry}

                if status_str == "error":
                    error_msg = ""
                    outputs = entry.get("outputs", {})
                    for node_id, node_out in outputs.items():
                        if "error" in node_out:
                            error_msg = str(node_out["error"])
                            break
                    if not error_msg:
                        error_msg = json.dumps(status.get("messages", []), ensure_ascii=False)
                    logger.error("Prompt %s failed: %s", prompt_id, error_msg)
                    error("ComfyUIAdapter", "Prompt %s 失败: %s", prompt_id, error_msg[:200])
                    return {"status": "error", "prompt_id": prompt_id, "error": error_msg}

            time.sleep(poll_interval)

    def get_output_video_path(self, prompt_id):
        try:
            url = f"{self._base_url}/history/{prompt_id}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                history = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return ""

        entry = history.get(prompt_id, {})
        outputs = entry.get("outputs", {})

        for node_id, node_out in outputs.items():
            videos = node_out.get("videos", [])
            if videos:
                for v in videos:
                    fname = v.get("filename", "")
                    subfolder = v.get("subfolder", "")
                    if fname:
                        return f"{subfolder}/{fname}" if subfolder else fname

            gifs = node_out.get("gifs", [])
            if gifs:
                for g in gifs:
                    fname = g.get("filename", "")
                    subfolder = g.get("subfolder", "")
                    if fname:
                        return f"{subfolder}/{fname}" if subfolder else fname

            images = node_out.get("images", [])
            if images:
                for img in images:
                    fname = img.get("filename", "")
                    subfolder = img.get("subfolder", "")
                    if fname:
                        return f"{subfolder}/{fname}" if subfolder else fname

        return ""

    def discover_nodes(self, workflow):
        node_map = NodeMap()

        for nid, node in workflow.items():
            class_type = node.get("class_type", "")

            if class_type == "WanVideoAnimateEmbeds":
                node_map.animate_embeds = nid
            elif class_type == "VHS_VideoCombine":
                node_map.video_combine = nid
            elif class_type == "LoadImage":
                if not node_map.ref_image:
                    node_map.ref_image = nid
            elif class_type == "VHS_LoadVideo":
                node_map.ref_video = nid
            elif class_type == "WanVideoSampler":
                node_map.sampler = nid
            elif class_type == "WanVideoTextEncode":
                node_map.text_encode = nid

        for nid, node in workflow.items():
            class_type = node.get("class_type", "")
            if class_type in ("WanVideoAnimateEmbeds",) and not node_map.pose_images:
                inputs = node.get("inputs", {})
                if "pose_images" in inputs:
                    node_map.pose_images = nid

        return node_map

    def modify_workflow_for_segment(self, wf, node_map, seg, ref_image_path, pose_dir="", run_id=""):
        import folder_paths
        import shutil

        wf = json.loads(json.dumps(wf))

        if ref_image_path and node_map.ref_image:
            img_name = self._copy_to_input(ref_image_path)
            if img_name and node_map.ref_image in wf:
                wf[node_map.ref_image]["inputs"]["image"] = img_name

        if node_map.animate_embeds and node_map.animate_embeds in wf:
            ae = wf[node_map.animate_embeds]
            ae["inputs"]["width"] = seg.params.get("width", 832)
            ae["inputs"]["height"] = seg.params.get("height", 480)
            ae["inputs"]["num_frames"] = seg.target_frames

            if pose_dir and "pose_images" not in ae.get("inputs", {}):
                pose_video = os.path.join(pose_dir, "poses.mp4")
                if os.path.isfile(pose_video):
                    pose_node_id = "yunjii_pose_video"
                    wf[pose_node_id] = {
                        "class_type": "VHS_LoadVideo",
                        "inputs": {
                            "video": pose_video,
                            "force_rate": 0,
                            "force_size": "Disabled",
                            "custom_width": 512,
                            "custom_height": 512,
                            "frame_start": 0,
                            "frame_load_cap": seg.target_frames,
                            "skip_first_frames": 0,
                            "select_every_nth": 1,
                            "choose video to upload": "video",
                        },
                    }
                    ae["inputs"]["pose_images"] = [pose_node_id, 0]

        if node_map.sampler and node_map.sampler in wf:
            sampler = wf[node_map.sampler]
            if "steps" in sampler.get("inputs", {}):
                sampler["inputs"]["steps"] = seg.params.get("steps", 30)
            if "cfg" in sampler.get("inputs", {}):
                sampler["inputs"]["cfg"] = seg.params.get("cfg", 6.0)

        if node_map.text_encode and node_map.text_encode in wf:
            te = wf[node_map.text_encode]
            if "positive_prompt" in te.get("inputs", {}):
                te["inputs"]["positive_prompt"] = seg.prompt
            elif "prompt" in te.get("inputs", {}):
                te["inputs"]["prompt"] = seg.prompt
            if "negative_prompt" in te.get("inputs", {}) and seg.negative_prompt:
                te["inputs"]["negative_prompt"] = seg.negative_prompt

        return wf

    def _copy_to_input(self, src_path):
        try:
            import folder_paths
            import shutil
            input_dir = folder_paths.get_input_directory()
            os.makedirs(input_dir, exist_ok=True)
            fname = os.path.basename(src_path)
            dst = os.path.join(input_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
            return fname
        except Exception as e:
            logger.error("Failed to copy to input: %s", e)
            return ""

    def interrupt(self):
        try:
            req = urllib.request.Request(
                f"{self._base_url}/interrupt",
                data=b"",
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
