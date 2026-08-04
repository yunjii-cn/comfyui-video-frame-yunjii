"""
ComfyUI Video Frame Processing Plugin
Yunjii (云集智能)
Copyright 2026
"""
import os
import sys
import importlib
import traceback
import logging

logger = logging.getLogger("yunjii")

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
from .engine.planner import YunjiiSegmentPlanner
from .engine.runner import YunjiiSegmentRunner
from .engine.stitcher import YunjiiSegmentStitcher
from .engine.composer import YunjiiVideoImitator
from .engine.latent_nodes import YunjiiSaveLatent, YunjiiLoadLatent

_detection_dir = os.path.join(os.path.dirname(_comfyui_root), "models", "detection")
if not os.path.isdir(_detection_dir):
    _detection_dir = os.path.join(_comfyui_root, "models", "detection")
if os.path.isdir(_detection_dir):
    try:
        folder_paths.add_model_folder_path("detection", _detection_dir)
    except Exception:
        pass

NODE_CLASS_MAPPINGS.update({
    "YunjiiSegmentPlanner": YunjiiSegmentPlanner,
    "YunjiiSegmentRunner": YunjiiSegmentRunner,
    "YunjiiSegmentStitcher": YunjiiSegmentStitcher,
    "YunjiiVideoImitator": YunjiiVideoImitator,
    "YunjiiSaveLatent": YunjiiSaveLatent,
    "YunjiiLoadLatent": YunjiiLoadLatent,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "YunjiiSegmentPlanner": "分段规划 🧠 (Yunjii V2V)",
    "YunjiiSegmentRunner": "链式执行 ⛓ (Yunjii V2V)",
    "YunjiiSegmentStitcher": "无缝拼接 🎞 (Yunjii V2V)",
    "YunjiiVideoImitator": "完美模仿一键 ⚡ (Yunjii V2V)",
    "YunjiiSaveLatent": "Latent 落盘 💾 (Yunjii V2V)",
    "YunjiiLoadLatent": "Latent 读取 📂 (Yunjii V2V)",
})

WEB_DIRECTORY = "./js"

_PLUGIN_PACKAGE = __name__
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

_YUNJII_NODE_NAMES = set(NODE_CLASS_MAPPINGS.keys())


def _clear_module_cache():
    cleared = []
    preserve = _PLUGIN_PACKAGE
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(_PLUGIN_PACKAGE + "."):
            del sys.modules[mod_name]
            cleared.append(mod_name)
    return cleared


def _reload_plugin():
    errors = []
    cleared = _clear_module_cache()

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            _PLUGIN_PACKAGE,
            os.path.join(_PLUGIN_DIR, "__init__.py"),
            submodule_search_locations=[_PLUGIN_DIR],
        )
        plugin_mod = importlib.util.module_from_spec(spec)
        sys.modules[_PLUGIN_PACKAGE] = plugin_mod
        spec.loader.exec_module(plugin_mod)
    except Exception as e:
        errors.append(f"Failed to import plugin: {e}\n{traceback.format_exc()}")
        return {"status": "error", "errors": errors, "cleared": len(cleared)}

    new_mappings = getattr(plugin_mod, "NODE_CLASS_MAPPINGS", {})
    new_display = getattr(plugin_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})

    try:
        import nodes as comfy_nodes
        for name, node_cls in new_mappings.items():
            comfy_nodes.NODE_CLASS_MAPPINGS[name] = node_cls
        comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(new_display)
    except Exception as e:
        errors.append(f"Failed to update ComfyUI registry: {e}")

    reloaded_count = len(cleared)
    node_count = len(new_mappings)

    return {
        "status": "ok" if not errors else "partial",
        "reloaded_modules": reloaded_count,
        "node_count": node_count,
        "nodes": list(new_mappings.keys()),
        "errors": errors,
    }


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

        @server.PromptServer.instance.routes.post("/yunjii/reload")
        async def yunjii_reload(request):
            try:
                result = _reload_plugin()
                logger.info("[Yunjii] Hot reload: %s", result)
                return web.json_response(result)
            except Exception as e:
                logger.error("[Yunjii] Hot reload failed: %s", e)
                return web.json_response({"status": "error", "error": str(e)})

        @server.PromptServer.instance.routes.get("/yunjii/logs")
        async def yunjii_logs(request):
            try:
                from .engine.debug_log import list_recent_logs, read_log, get_current_log_path

                filename = request.query.get("file", "")
                tail = int(request.query.get("tail", "100"))

                if filename:
                    content = read_log(filename, tail_lines=tail)
                    return web.json_response({"file": filename, "content": content})

                recent = list_recent_logs(20)
                current = os.path.basename(get_current_log_path()) if get_current_log_path() else ""
                return web.json_response({"recent_logs": recent, "current": current})
            except Exception as e:
                return web.json_response({"error": str(e)})

        @server.PromptServer.instance.routes.get("/yunjii/status")
        async def yunjii_status(request):
            try:
                from .engine.debug_log import get_current_log_path, get_log_dir_path
                return web.json_response({
                    "plugin": "comfyui-video-frame-yunjii",
                    "log_dir": get_log_dir_path(),
                    "current_log": get_current_log_path(),
                    "nodes": list(NODE_CLASS_MAPPINGS.keys()),
                    "node_count": len(NODE_CLASS_MAPPINGS),
                })
            except Exception as e:
                return web.json_response({"error": str(e)})

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
