"""把视频登记进 ComfyUI 前端历史，使其出现在「已生成 / 画廊」。

与 `DirectAdapter._register_frontend` 同源逻辑：按 VHS_VideoCombine 的 gifs/videos/images
schema 写入 `pq.history` 并广播 `executed` / `execution_success`，让前端即时显示。

为何需要独立登记（而非仅靠节点 return ui）：
    `YunjiiVideoImitator`(完美模仿) 在节点内部通过 `DirectAdapter` 内联执行各段生成
    （`executor.execute_async`）。内联执行会改写外层 prompt 的历史上下文，导致 Imitator
    节点返回的 `ui` 不被外层 `history[outer_prompt_id]` 捕获——最终成片因此不出现在
    「已生成」。段视频之所以能显示，正是因为它们走了本模块同款的手动登记通道。
    故最终成片也在此显式登记一条独立历史条目，确保稳定可见（与段视频并列，互不干扰）。
"""
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)


def register_video_to_history(video_path: str, workflow_dict: dict = None,
                              output_node_id: str = "YunjiiVideoImitator") -> bool:
    """把单个最终视频登记为一条独立的前端历史(已生成)记录。成功返回 True。

    - 使用全新 prompt_id，避免被外层图执行结束时的 history 写回覆盖。
    - schema 1:1 对齐 VHS_VideoCombine（gifs/videos/images），最大化前端渲染兼容。
    - 失败仅告警并返回 False，绝不抛异常影响出片主流程。
    """
    if not video_path or not os.path.isfile(video_path):
        logger.warning("register_video_to_history: 视频不存在，跳过登记: %s", video_path)
        return False
    try:
        import folder_paths
        from server import PromptServer
        server = PromptServer.instance

        out_dir = folder_paths.get_output_directory()
        fname = os.path.basename(video_path)
        d = os.path.dirname(video_path)
        rel = os.path.relpath(d, out_dir) if d.startswith(out_dir) else ""
        # 前端 /view 走 URL 查询参数(?subfolder=...)，必须正斜杠（Windows 下 relpath 返反斜杠→404）。
        subfolder = "" if rel in (".", "") else rel.replace(os.sep, "/")

        # 实际 fps 覆盖兜底值；视频组件缺 frame_rate 字段时不渲染。
        _fps = None
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            _fps = _cap.get(_cv2.CAP_PROP_FPS)
            _cap.release()
        except Exception:
            pass

        # 首帧封面：ffmpeg 稳健抽取（画廊只认 images；cv2 在 ffmpeg 编码成片上可能失败）。
        _first = ""
        try:
            from .poster import extract_poster_png
            _first = extract_poster_png(video_path)
        except Exception:
            _first = ""

        preview = {
            "filename": fname,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/h264-mp4",
            "frame_rate": float(_fps) if _fps and _fps > 0 else 16.0,
            "workflow": os.path.basename(_first) if _first else None,
            "fullpath": video_path,
        }
        images = []
        if _first and os.path.isfile(_first):
            images.append({"filename": os.path.basename(_first),
                           "subfolder": subfolder, "type": "output"})

        node_ui = {"gifs": [preview], "videos": [preview], "images": images}
        prompt_id = str(uuid.uuid4())
        outputs = {output_node_id: node_ui}

        pq = getattr(server, "prompt_queue", None)
        if pq is not None and hasattr(pq, "history"):
            now_ms = int(time.time() * 1000)
            extra_data = {"client_id": None, "extra_pnginfo": {}}
            # HeiHe fork 的 normalize_history_item 期望 history['prompt'] 是 5 元组
            # (priority, _, prompt_dict, extra_data, _)，否则 get_jobs 解包报
            # "too many values to unpack (expected 5)" 使前端历史拉取崩溃。
            pq.history[prompt_id] = {
                "prompt": (0, None, workflow_dict or {}, extra_data, None),
                "outputs": outputs,
                "status": {
                    "status_str": "success",
                    "messages": [
                        ["execution_start", {"prompt_id": prompt_id, "timestamp": now_ms}],
                        ["execution_success", {"prompt_id": prompt_id, "timestamp": now_ms}],
                    ],
                },
            }
            try:
                server.queue_updated()
            except Exception:
                pass

        # 实时广播：execution_success 让 Gallery/Queue-History 画廊刷新；
        # executed 的 output 必须是「单个节点」的 ui 字典，使画廊正确渲染。
        try:
            server.send_sync("execution_success", {"prompt_id": prompt_id}, None)
        except Exception:
            pass
        try:
            server.send_sync("executed", {
                "prompt_id": prompt_id,
                "node": output_node_id,
                "display_node": output_node_id,
                "output": node_ui,
            }, None)
        except Exception:
            pass

        logger.info("已登记最终成片到前端历史 prompt_id=%s file=%s subfolder=%s",
                    prompt_id, fname, subfolder)
        return True
    except Exception as e:
        logger.warning("登记最终成片到前端历史失败(不影响出片): %s", e)
        return False
