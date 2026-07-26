import logging

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
