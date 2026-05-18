"""
ComfyUI Video Frame Processing Plugin
Yunjii (云集智能)
Copyright 2026
"""
import os
import sys

_comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_comfy_utils = os.path.join(_comfyui_root, "comfy", "utils")
if os.path.isdir(_comfy_utils):
    import importlib.util
    _existing = sys.modules.get("utils")
    _need_fix = False
    if _existing is None:
        _need_fix = True
    elif not hasattr(_existing, "__path__") or not isinstance(getattr(_existing, "__path__", None), list):
        _need_fix = True
    if _need_fix:
        _spec = importlib.util.spec_from_file_location("utils", os.path.join(_comfy_utils, "__init__.py"), submodule_search_locations=[_comfy_utils])
        _utils_mod = importlib.util.module_from_spec(_spec)
        sys.modules["utils"] = _utils_mod
        _spec.loader.exec_module(_utils_mod)

import folder_paths
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"


def _register_upload_route():
    try:
        import server
        from aiohttp import web

        @server.PromptServer.instance.routes.post("/yunjii/upload_video")
        async def yunjii_upload_video(request):
            try:
                input_dir = folder_paths.get_input_directory()
                os.makedirs(input_dir, exist_ok=True)
                reader = await request.multipart()
                async for part in reader:
                    if part.name not in ("file", "files[]", "files"):
                        continue
                    filename = part.filename or ""
                    if not filename:
                        continue
                    safe_name = _safe_upload_name(input_dir, filename, ".mp4")
                    dst_path = os.path.join(input_dir, safe_name)
                    with open(dst_path, "wb") as f:
                        while True:
                            chunk = await part.read_chunk(1024 * 256)
                            if not chunk:
                                break
                            f.write(chunk)
                    return web.json_response({"saved": safe_name})
                return web.json_response({"saved": "", "error": "未收到文件"})
            except Exception as e:
                return web.json_response({"saved": "", "error": str(e)})
    except ImportError:
        pass


def _safe_upload_name(input_dir, original_name, default_ext):
    base = os.path.basename(str(original_name or "")).strip()
    if not base:
        base = f"upload{default_ext}"
    base = base.replace("\\", "_").replace("/", "_")
    name, ext = os.path.splitext(base)
    if not ext:
        ext = default_ext
    safe = f"yunjii_{name}{ext}"
    dst = os.path.join(input_dir, safe)
    if not os.path.exists(dst):
        return safe
    import time
    return f"yunjii_{name}_{int(time.time())}{ext}"


_register_upload_route()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
