import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, mode="一镜到底"):
        self.mode = mode
        self._dir = ""

    def _get_dir(self):
        if not self._dir:
            try:
                import folder_paths
                self._dir = os.path.join(folder_paths.get_output_directory(), "yunjii_checkpoints")
            except Exception:
                self._dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
            os.makedirs(self._dir, exist_ok=True)
        return self._dir

    def _get_path(self, run_id=None):
        d = self._get_dir()
        if run_id:
            return os.path.join(d, f"run_{run_id}.json")
        checkpoints = sorted(
            [f for f in os.listdir(d) if f.startswith("run_") and f.endswith(".json")],
            reverse=True,
        )
        if checkpoints:
            return os.path.join(d, checkpoints[0])
        return os.path.join(d, f"run_{int(time.time())}.json")

    def save(self, current_segment, prev_last_frame, results):
        data = {
            "mode": self.mode,
            "current_segment": current_segment,
            "prev_last_frame": prev_last_frame,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self._get_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Checkpoint saved: segment %d", current_segment)
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)

    def load(self):
        path = self._get_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Checkpoint loaded: segment %d", data.get("current_segment", 0))
            return data
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return None

    def clear(self):
        path = self._get_path()
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass
