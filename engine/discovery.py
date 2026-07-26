import logging

from .types import NodeMap

logger = logging.getLogger(__name__)

NODE_TYPE_MAP = {
    "animate_embeds": [
        "WanVideoAnimateEmbeds",
    ],
    "video_combine": [
        "VHS_VideoCombine",
    ],
    "ref_image": [
        "LoadImage",
    ],
    "ref_video": [
        "VHS_LoadVideo",
    ],
    "sampler": [
        "WanVideoSampler",
    ],
    "text_encode": [
        "WanVideoTextEncode",
    ],
}


def auto_discover_nodes(workflow):
    node_map = NodeMap()
    assigned = set()

    for role, class_types in NODE_TYPE_MAP.items():
        for nid, node in workflow.items():
            if nid in assigned:
                continue
            ct = node.get("class_type", "")
            if ct in class_types:
                setattr(node_map, role, nid)
                assigned.add(nid)
                break

    if not node_map.ref_image:
        for nid, node in workflow.items():
            if nid in assigned:
                continue
            if node.get("class_type", "") == "LoadImage":
                node_map.ref_image = nid
                assigned.add(nid)
                break

    logger.info("Discovered nodes: %s", node_map.to_dict())
    return node_map
