import os
import uuid
import json
import time
import threading
import asyncio
import traceback
from typing import Optional

from .base import GenerationAdapter
from ..types import NodeMap
from ..debug_log import info, warn, error as log_error


class DirectAdapter(GenerationAdapter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._last_video_path: Optional[str] = None
        self._persistent_executor = None
        self._accel_lora_applied = False

    def init_executor(self):
        from execution import PromptExecutor, CacheType
        from server import PromptServer

        server = PromptServer.instance
        self._persistent_executor = PromptExecutor(
            server, cache_type=CacheType.RAM_PRESSURE, cache_args=self._build_cache_args()
        )
        info("DirectAdapter", "初始化持久化执行器 (RAM_PRESSURE缓存, 8GB headroom, T5/模型输出跨段复用)")

    @staticmethod
    def _build_cache_args() -> dict:
        """复刻 ComfyUI main.py 的 RAM_PRESSURE 缓存参数构造。

        秋叶整合包定制版 execution.execute_async 直接按下标取
        cache_args['ram'] 与 cache_args['ram_inactive']，二者缺一不可，
        缺 'ram_inactive' 会抛 KeyError（见此前报错）。
        """
        try:
            import comfy.model_management as mm
            total_ram_gb = mm.total_ram / 1024.0  # total_ram 单位为 MB
        except Exception:
            total_ram_gb = 128.0  # 兜底：按本机 128GB 估算
        ram_inactive = min(96.0, total_ram_gb)
        return {"ram": 8.0, "ram_inactive": ram_inactive}

    def cleanup_executor(self):
        if self._persistent_executor is not None:
            del self._persistent_executor
        self._persistent_executor = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info("DirectAdapter", "清理持久化执行器, 释放缓存+GPU显存")

    def submit(self, workflow_dict: dict) -> str:
        return str(uuid.uuid4())

    def wait(self, prompt_id: str, timeout: int = 1800) -> dict:
        return {"status": "submitted"}

    def execute_inline(self, workflow_dict: dict, timeout: int = 1800, primary_output_node=None) -> dict:
        result_container = {}
        exec_start_time = time.time()

        def worker():
            try:
                from execution import PromptExecutor, CacheType
                from server import PromptServer

                server = PromptServer.instance
                saved_client_id = server.client_id
                server.client_id = None

                use_persistent = self._persistent_executor is not None

                if use_persistent:
                    executor = self._persistent_executor
                    cache_mode = "RAM_PRESSURE缓存(持久化执行器)"
                else:
                    cache_args = self._build_cache_args()
                    executor = PromptExecutor(server, cache_type=CacheType.RAM_PRESSURE, cache_args=cache_args)
                    cache_mode = "RAM_PRESSURE缓存(临时执行器)"

                prompt_id = str(uuid.uuid4())

                info("DirectAdapter", "开始内联执行 prompt_id=%s (%s)", prompt_id, cache_mode)

                extra_data = {"client_id": "yunjii_direct"}

                execute_outputs = self._get_output_node_ids(workflow_dict, primary_output_node)
                info("DirectAdapter", "execute_outputs=%s (primary=%s)", execute_outputs, primary_output_node)

                if not execute_outputs:
                    log_error("DirectAdapter", "未找到输出节点! 工作流中的节点: %s",
                              list(workflow_dict.keys()))
                    result_container["status"] = "error"
                    result_container["error"] = "工作流中未找到输出节点(OUTPUT_NODE=True)"
                    result_container["prompt_id"] = prompt_id
                    return

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        executor.execute_async(workflow_dict, prompt_id, extra_data, execute_outputs)
                    )
                finally:
                    loop.close()

                server.client_id = saved_client_id

                exec_duration = time.time() - exec_start_time
                info("DirectAdapter", "执行耗时=%.2fs success=%s", exec_duration, executor.success)

                if not executor.success:
                    error_detail = self._extract_error_from_executor(executor, workflow_dict)
                    log_error("DirectAdapter", "节点报错详情: %s", error_detail)
                    result_container["status"] = "error"
                    result_container["error"] = f"工作流执行失败: {error_detail}"
                    result_container["prompt_id"] = prompt_id
                    return

                video_path = self._extract_video_from_history(executor, prompt_id, primary_output_node)
                if not video_path:
                    video_path = self._find_video_after(exec_start_time)

                if video_path and os.path.isfile(video_path):
                    self._last_video_path = video_path
                    result_container["status"] = "success"
                    result_container["video_path"] = video_path
                    result_container["prompt_id"] = prompt_id
                    # 方案A(自动化)：把结果注册进前端历史，让资产出现在「已生成/历史」里。
                    # 内联执行绕过了 API 队列，默认不写历史->前端看不到；这里手动补齐，
                    # 同时保留内联的程序化控制力(分段/重试/续跑)。
                    try:
                        self._register_frontend(server, prompt_id, video_path, workflow_dict, execute_outputs)
                    except Exception as _reg_e:
                        warn("DirectAdapter", "前端历史注册异常(不影响出片): %s", _reg_e)
                else:
                    history = getattr(executor, 'history_result', None)
                    info("DirectAdapter", "未找到视频: history=%s, 扫描=%s",
                         json.dumps(history, default=str)[:500] if history else "None",
                         self._find_video_after(exec_start_time) or "空")

                    executed_nodes = getattr(executor, 'status_messages', [])
                    executed_node_ids = set()
                    for ev, data in executed_nodes:
                        if ev == "executing" and "node" in data:
                            executed_node_ids.add(data["node"])
                    info("DirectAdapter", "已执行节点: %s", executed_node_ids)

                    result_container["status"] = "error"
                    result_container["error"] = "生成完成但未找到输出视频文件"
                    result_container["prompt_id"] = prompt_id

                info("DirectAdapter", "内联执行完成 prompt_id=%s status=%s", prompt_id, result_container.get("status"))

            except Exception as exc:
                tb = traceback.format_exc()
                log_error("DirectAdapter", "执行异常: %s\n%s", exc, tb)
                result_container["status"] = "error"
                result_container["error"] = f"执行异常: {exc}"

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # 超时：daemon 线程仍在持久化 executor 上执行。
            # ComfyUI 的模型加载/GPU 是全局共享的（comfy.model_management），若直接返回并让调用方重试，
            # 残留线程会复用/新建 executor 并与之并发执行 -> 争用同一显存与已加载模型，结果损坏。
            # 因此必须先打断残留执行并等其真正结束，再让下一轮重试在独立的临时 executor 上进行。
            log_error("DirectAdapter", "执行超时(%ss)，尝试中断残留执行线程", timeout)
            try:
                import nodes as _nodes
                _nodes.interrupt_processing(False)
            except Exception as _e:
                log_error("DirectAdapter", "interrupt_processing 调用失败(忽略): %s", _e)
            # 中断信号下等待残留线程自然结束（最多再等 120s 兜底，避免极端死锁时永久挂起）
            t.join(timeout=120)
            if t.is_alive():
                warn("DirectAdapter", "残留线程在中断后仍存活，强行放弃持久化执行器（下一次将新建临时执行器）")
            # 放弃持久化执行器：下一次 execute_inline 会新建独立临时执行器，隔离任何残留状态
            self._persistent_executor = None
            return {"status": "timeout", "error": f"超时 ({timeout}s)"}

        if not result_container:
            log_error("DirectAdapter", "worker线程已结束但result_container为空, 可能CUDA OOM")
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"status": "error", "error": "执行异常: worker线程结束但无结果(可能CUDA OOM)"}

        return result_container

    @staticmethod
    def _extract_error_from_executor(executor, workflow_dict: dict) -> str:
        status_messages = getattr(executor, 'status_messages', [])
        for event, data in reversed(status_messages):
            if event == "execution_error":
                node_id = data.get("node_id", "?")
                node_type = data.get("node_type", "?")
                msg = data.get("exception_message", "未知错误")
                exc_type = data.get("exception_type", "")
                tb_lines = data.get("traceback", [])
                tb_str = "".join(tb_lines[:3]) if tb_lines else ""
                class_type = "?"
                if node_id in workflow_dict:
                    class_type = workflow_dict[node_id].get("class_type", "?")
                return f"节点{node_id}({class_type}/{node_type}): [{exc_type}] {msg}\n{tb_str}"
            elif event == "execution_interrupted":
                node_id = data.get("node_id", "?")
                return f"执行被中断(节点{node_id})"
        return "未知错误(无execution_error消息)"

    @staticmethod
    def _get_output_node_ids(workflow_dict, primary_output_node=None):
        output_nodes = []
        try:
            import nodes
            for node_id, node_data in workflow_dict.items():
                if not isinstance(node_data, dict):
                    continue
                class_type = node_data.get("class_type", "")
                node_cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
                if node_cls is not None:
                    if getattr(node_cls, 'OUTPUT_NODE', False):
                        output_nodes.append(node_id)
        except Exception:
            for node_id, node_data in workflow_dict.items():
                if isinstance(node_data, dict):
                    ct = node_data.get("class_type", "")
                    if any(kw in ct for kw in ["VideoCombine", "SaveImage", "PreviewImage"]):
                        output_nodes.append(node_id)
        # 优先返回调用方指定的主输出节点（Runner 控制的真实成片 VHS），
        # 避免多输出节点时抽到姿态/预览类骨架视频。
        # 注意：discover 读 UI 格式(id 为 int)，而 API prompt/history 用字符串键，
        # 故统一转 str 比对，保证主节点能被正确识别。
        primary = str(primary_output_node) if primary_output_node is not None else None
        if primary and primary in [str(o) for o in output_nodes]:
            return [primary]
        return output_nodes

    def _extract_video_from_history(self, executor, prompt_id: str, primary_output_node=None) -> str:
        history = getattr(executor, 'history_result', None)
        if not history:
            return ""

        outputs = history.get("outputs", {})
        if not outputs:
            return ""

        # 排序：主输出节点优先（统一转 str，兼容 UI int id 与 API 字符串键）
        primary = str(primary_output_node) if primary_output_node is not None else None
        ordered = []
        if primary and primary in outputs:
            ordered.append(primary)
        ordered += [str(n) for n in outputs if str(n) not in ordered]

        real_candidates = []   # 非姿态/骨架的真实成片
        any_candidates = []    # 兜底（含姿态）
        for nid in ordered:
            node_output = outputs[nid]
            path, sub = self._scan_one_output(node_output)
            if not path:
                continue
            any_candidates.append(path)
            # 硬性排除姿态/骨架预览节点（如 onetotall_pose_*.mp4），绝不返回骨架视频
            if DirectAdapter._is_pose_output(path):
                continue
            real_candidates.append((path, sub))

        # 优先返回带真实前缀的成片（yunjii_v2v 或 WanVideo_SCAIL），覆盖两条生成路线
        for path, sub in real_candidates:
            if "yunjii_v2v" in (sub or "") or "WanVideo_SCAIL" in os.path.basename(path):
                return path
        if real_candidates:
            return real_candidates[0][0]
        # 兜底：实在没有非姿态输出时，退回第一个（理论上不会走到）
        if any_candidates:
            return any_candidates[0]
        return ""

    def _scan_one_output(self, node_output):
        """从单个节点的历史输出里抽取视频路径。返回 (path, subfolder)。"""
        if not isinstance(node_output, dict):
            return ("", "")
        for key in ("gifs", "videos"):
            for info in node_output.get(key, []):
                if not isinstance(info, dict):
                    continue
                fullpath = info.get("fullpath", "")
                if fullpath and os.path.isfile(fullpath):
                    return (fullpath, info.get("subfolder", ""))
                filename = info.get("filename", "")
                subfolder = info.get("subfolder", "")
                file_type = info.get("type", "output")
                if filename:
                    resolved = self._resolve_output_file(filename, subfolder, file_type)
                    if resolved:
                        return (resolved, subfolder)
        for info in node_output.get("images", []):
            if not isinstance(info, dict):
                continue
            filename = info.get("filename", "")
            subfolder = info.get("subfolder", "")
            file_type = info.get("type", "output")
            if filename and (filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.gif')) or file_type == "output"):
                resolved = self._resolve_output_file(filename, subfolder, file_type)
                if resolved:
                    return (resolved, subfolder)
        return ("", "")

    def _register_frontend(self, server, prompt_id, video_path, workflow_dict, execute_outputs):
        """把内联执行的结果注册进前端历史，让资产出现在「已生成/历史」里。

        内联执行绕过了 API 队列，默认不写 prompt_queue.history，因此前端画廊/历史
        看不到产出。这里手动补齐：按 VHS_VideoCombine 的 gifs 输出格式写入历史，并
        广播 executed / queue_updated，使前端即时显示。失败仅告警，不影响出片。
        """
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
            fname = os.path.basename(video_path)
            d = os.path.dirname(video_path)
            rel = os.path.relpath(d, out_dir) if d.startswith(out_dir) else ""
            subfolder = "" if rel in (".", "") else rel.replace(os.sep, "/")

            # 对齐 VHS 黄金标准：video/h264-mp4 + 实际 fps + 首帧封面 + 同时返回 gifs/videos，
            # 与 _build_output_ui 保持完全一致，避免画廊/历史里视频同样不渲染。
            _fps = None
            _first = ""
            try:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(video_path)
                _fps = _cap.get(_cv2.CAP_PROP_FPS)
                _r, _frm = _cap.read()
                _cap.release()
                if _r and _frm is not None:
                    _fs = os.path.splitext(fname)[0]
                    _first = os.path.join(os.path.dirname(video_path), _fs + "_first.png")
                    try:
                        _cv2.imwrite(_first, _frm)
                    except Exception:
                        _first = ""
            except Exception:
                pass
            preview = {
                "filename": fname,
                "subfolder": subfolder,
                "type": "output",
                "format": "video/h264-mp4",
                "frame_rate": float(_fps) if _fps and _fps > 0 else 16.0,
                "workflow": os.path.basename(_first) if _first else None,
                "fullpath": video_path,
            }
            node_ui = {"gifs": [preview], "videos": [preview]}
            outputs = {}
            for onid in execute_outputs:
                outputs[onid] = node_ui

            pq = getattr(server, "prompt_queue", None)
            if pq is not None and hasattr(pq, "history"):
                import time as _time
                now_ms = int(_time.time() * 1000)
                extra_data = {"client_id": None, "extra_pnginfo": {}}
                # 重要：本 HeiHe fork 的 normalize_history_item 期望 history['prompt']
                # 是 5 元组 (priority, _, prompt_dict, extra_data, _)。若直接传 dict，
                # get_jobs 解包会报 "too many values to unpack (expected 5)"，导致
                # 前端「已生成/历史」面板拉取历史时崩溃。务必按 5 元组写入。
                pq.history[prompt_id] = {
                    "prompt": (0, None, workflow_dict, extra_data, None),
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
            # 实时广播：executed 仅更新画布节点预览；真正让 Gallery/Queue-History
            # 画廊刷新的信号是 execution_success（配合前端「Auto-refresh after
            # generation」开关，本 fork 已默认打开）。两者都发，确保内联出片后
            # 视频立即出现在前端资产画廊，而无需手动刷新/重载页面。
            try:
                server.send_sync("execution_success", {
                    "prompt_id": prompt_id,
                }, None)
            except Exception:
                pass
            # 对照 execution.py:577-578 标准格式：executed 的 output 必须是「单个节点」的
            # ui 字典({"gifs":[...],"videos":[...]})，而非按节点分组的 history 结构。
            # 先前误发 outputs(={onid:{...}}) → 前端 msg.output.gifs 为 undefined → 节点/画廊全白。
            # 每个输出节点各发一条 executed，使画布预览与画廊都能正确渲染。
            for onid in execute_outputs:
                try:
                    server.send_sync("executed", {
                        "prompt_id": prompt_id,
                        "node": onid,
                        "display_node": onid,
                        "output": node_ui,
                    }, None)
                except Exception:
                    pass
            info("DirectAdapter", "已注册结果到前端历史 prompt_id=%s file=%s subfolder=%s",
                 prompt_id, fname, subfolder)
        except Exception as e:
            warn("DirectAdapter", "注册前端历史失败(不影响出片): %s", e)

    def _resolve_output_file(self, filename: str, subfolder: str = "", file_type: str = "output") -> str:
        import folder_paths

        if file_type == "temp":
            base_dir = folder_paths.get_temp_directory()
        else:
            base_dir = folder_paths.get_output_directory()

        if subfolder:
            path = os.path.join(base_dir, subfolder, filename)
        else:
            path = os.path.join(base_dir, filename)

        if os.path.isfile(path):
            return path

        output_dir = folder_paths.get_output_directory()
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            return path

        if file_type != "temp":
            temp_dir = folder_paths.get_temp_directory()
            path = os.path.join(temp_dir, filename)
            if os.path.isfile(path):
                return path

        return ""

    def _find_video_after(self, after_time: float) -> str:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        if not output_dir:
            return ""
        candidates = []
        try:
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                        ftime = os.path.getmtime(fp)
                        if ftime >= after_time:
                            candidates.append((ftime, fp))
        except Exception:
            return ""
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return ""

    def extract_frame(self, video_path: str, frame_idx: int) -> Optional[str]:
        if not os.path.isfile(video_path):
            return None
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = max(0, min(frame_idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        out_path = os.path.join(self.output_dir, f"segment_ref_{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(out_path, frame)
        return out_path

    # 姿态/骨架预览类输出的文件名标记（用于历史抽取时硬性排除，避免抽到骨架视频）
    POSE_PREFIX_MARKERS = ("pose", "skeleton", "openpose", "dwpose", "onetotall", "nlfpose")

    @staticmethod
    def _vhs_meta(ndata):
        """读取一个 VHS_VideoCombine 节点的 (文件名前缀, 是否 save_output)。
        同时兼容 UI 保存格式(widgets_values 为 dict) 与 API prompt 格式(inputs 为 dict)。"""
        wv = ndata.get("widgets_values")
        if isinstance(wv, dict):
            return wv.get("filename_prefix", ""), bool(wv.get("save_output", False))
        inp = ndata.get("inputs", {})
        if isinstance(inp, dict):
            return inp.get("filename_prefix", ""), bool(inp.get("save_output", False))
        return "", False

    @staticmethod
    def _is_pose_output(path_or_prefix):
        name = os.path.basename(str(path_or_prefix or "")).lower()
        return any(m in name for m in DirectAdapter.POSE_PREFIX_MARKERS)

    @staticmethod
    def _select_primary_vhs(vhs_list):
        """从若干 VHS_VideoCombine 候选里挑出真实成片节点。
        vhs_list: [(nid, prefix, save_out), ...]
        规则：① 非姿态节点且 save_output=True 优先；② 其次非姿态节点；③ 兜底第一个。"""
        for nid, prefix, save_out in vhs_list:
            if not DirectAdapter._is_pose_output(prefix) and save_out:
                return nid
        for nid, prefix, save_out in vhs_list:
            if not DirectAdapter._is_pose_output(prefix):
                return nid
        return vhs_list[0][0] if vhs_list else ""

    @staticmethod
    def _iter_nodes(workflow):
        """统一遍历工作流的节点，兼容三种格式：
        ① UI 完整格式 nodes=list(元素含 id)；② UI/API 中间格式 nodes=dict(键=id)；
        ③ API prompt 顶层格式：{node_id: {class_type, inputs}}，无 nodes 包裹
           （prepare_workflow 返回的就是这种）。"""
        if not isinstance(workflow, dict):
            return []
        nodes = workflow.get("nodes", None)
        if isinstance(nodes, dict):
            return list(nodes.items())
        if isinstance(nodes, list):
            return [(n.get("id"), n) for n in nodes if isinstance(n, dict)]
        # 无 nodes 包裹：判断顶层是否就是节点字典（每个值是带 class_type/type 的节点）
        if nodes is None and workflow:
            first_val = next(iter(workflow.values()), None)
            if isinstance(first_val, dict) and ("class_type" in first_val or "type" in first_val):
                return list(workflow.items())
        return []

    def discover_nodes(self, workflow: dict) -> NodeMap:
        node_map = NodeMap()
        vhs_candidates = []
        for nid, ndata in DirectAdapter._iter_nodes(workflow):
            if not isinstance(ndata, dict):
                continue
            ct = ndata.get("class_type") or ndata.get("type", "")
            if "WanVideoAnimateEmbeds" in ct:
                node_map.animate_embeds = nid
            elif "VHS_VideoCombine" in ct:
                prefix, save_out = DirectAdapter._vhs_meta(ndata)
                vhs_candidates.append((nid, prefix, save_out))
            elif "LoadImage" in ct:
                if not node_map.ref_image:
                    node_map.ref_image = nid
            elif "WanVideoSampler" in ct:
                node_map.sampler = nid
            elif "WanVideoTextEncode" in ct:
                node_map.text_encode = nid
        # 正确选择真实成片节点（剔除姿态/骨架预览，优先 save_output=True）
        node_map.video_combine = DirectAdapter._select_primary_vhs(vhs_candidates)
        return node_map

    def get_output_path(self, workflow_dict: dict) -> str:
        return self._find_video_after(0)

    @staticmethod
    def _validate_prompt(workflow_dict):
        try:
            from execution import validate_prompt
            import asyncio
            prompt_id = str(uuid.uuid4())
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        validate_prompt(prompt_id, workflow_dict, None)
                    )
                    valid, errors, good_outputs, node_errors = future.result(timeout=30)
            else:
                valid, errors, good_outputs, node_errors = asyncio.run(
                    validate_prompt(prompt_id, workflow_dict, None)
                )
            return valid, errors, node_errors
        except TypeError:
            try:
                from execution import validate_prompt
                valid, errors, nodes_info = validate_prompt(workflow_dict)
                return valid, errors, nodes_info
            except Exception as e2:
                return False, [str(e2)], {}
        except Exception as e:
            return False, [str(e)], {}

    FP8_SCALED_T5_MODELS = {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "Wan\\umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "Wan\\\\umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    }
    FP8_SCALED_REPLACEMENT = "Wan\\umt5_xxl_fp16.safetensors"

    def _fix_fp8_scaled_t5(self, workflow: dict) -> dict:
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get("class_type", "")
            if class_type != "LoadWanVideoT5TextEncoder":
                continue
            inputs = node_data.get("inputs", {})
            model_name = inputs.get("model_name", "")
            if model_name in self.FP8_SCALED_T5_MODELS or "fp8" in model_name.lower() and "scaled" in model_name.lower():
                info("DirectAdapter", "自动修复T5模型: %s → %s", model_name, self.FP8_SCALED_REPLACEMENT)
                inputs["model_name"] = self.FP8_SCALED_REPLACEMENT
        return workflow

    ACCEL_LORA_480P = "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16"
    ACCEL_LORA_GENERAL = "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64"
    ACCEL_LORA_STEPS = 4
    ANIMATE_MODEL_PATTERNS = ["Animate", "animate"]
    ANIMATE_MIN_STEPS = 12
    ANIMATE_POSE_STRENGTH = 1.0
    ANIMATE_FACE_STRENGTH = 0.9
    ANIMATE_ACCEL_LORA_HIGH = "i2v_A14b_high_noise_lora_rank64_lightx2v_4step"
    ANIMATE_ACCEL_LORA_LOW = "i2v_A14b_low_noise_lora_rank64_lightx2v_4step"
    ANIMATE_ACCEL_STEPS = 8
    ANIMATE_ACCEL_CFG = 1.0
    ANIMATE_ACCEL_SCHEDULER = "euler"
    ANIMATE_ACCEL_SHIFT = 8.0
    ENHANCE_LORA_PATTERNS = ["Fun-14B-InP-HPS2.1", "Fun-14B-InP"]
    ENHANCE_LORA_STRENGTH = 0.7

    @staticmethod
    def _is_animate_model(workflow: dict) -> bool:
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoModelLoader":
                model_name = node_data.get("inputs", {}).get("model", "")
                if any(p in model_name for p in DirectAdapter.ANIMATE_MODEL_PATTERNS):
                    return True
        return False

    def _remove_i2v_lora(self, workflow: dict):
        model_node_id = None
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoModelLoader":
                model_node_id = node_id
                break
        if model_node_id is None:
            return

        inputs = workflow[model_node_id].get("inputs", {})
        if "lora" not in inputs:
            return

        lora_ref = inputs["lora"]
        if isinstance(lora_ref, list) and len(lora_ref) >= 1:
            lora_node_id = str(lora_ref[0])
            lora_node = workflow.get(lora_node_id, {})
            if isinstance(lora_node, dict) and lora_node.get("class_type") == "WanVideoLoraSelect":
                existing_lora = lora_node.get("inputs", {}).get("lora", "")
                is_i2v_distill = any(p in existing_lora for p in [self.ACCEL_LORA_480P, self.ACCEL_LORA_GENERAL])
                if is_i2v_distill:
                    del inputs["lora"]
                    if lora_node_id in workflow:
                        del workflow[lora_node_id]
                    info("DirectAdapter", "移除I2V蒸馏LoRA节点[%s]: %s (与Animate模型不兼容)", lora_node_id, existing_lora)

    def _inject_animate_accel_lora(self, workflow: dict) -> dict:
        import folder_paths
        available_loras = folder_paths.get_filename_list("loras")

        high_lora = None
        low_lora = None
        for avail in available_loras:
            if self.ANIMATE_ACCEL_LORA_HIGH in avail and high_lora is None:
                high_lora = avail
            if self.ANIMATE_ACCEL_LORA_LOW in avail and low_lora is None:
                low_lora = avail

        model_node_id = None
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoModelLoader":
                model_node_id = node_id
                break
        if model_node_id is None:
            return workflow

        inputs = workflow[model_node_id].get("inputs", {})

        if "lora" in inputs:
            lora_ref = inputs["lora"]
            if isinstance(lora_ref, list) and len(lora_ref) >= 1:
                lora_node_id = str(lora_ref[0])
                lora_node = workflow.get(lora_node_id, {})
                if isinstance(lora_node, dict) and lora_node.get("class_type") == "WanVideoLoraSelect":
                    existing_lora = lora_node.get("inputs", {}).get("lora", "")
                    is_animate_accel = any(p in existing_lora for p in [self.ANIMATE_ACCEL_LORA_HIGH, self.ANIMATE_ACCEL_LORA_LOW])
                    if is_animate_accel:
                        self._accel_lora_applied = True
                        info("DirectAdapter", "已有Animate加速LoRA: %s", existing_lora)
                        return workflow
                    is_i2v_distill = any(p in existing_lora for p in [self.ACCEL_LORA_480P, self.ACCEL_LORA_GENERAL])
                    if is_i2v_distill:
                        del inputs["lora"]
                        if lora_node_id in workflow:
                            del workflow[lora_node_id]
                        info("DirectAdapter", "移除I2V LoRA: %s (与Animate模型不兼容)", existing_lora)
                    else:
                        info("DirectAdapter", "已有用户LoRA(%s), 跳过Animate加速LoRA注入", existing_lora)
                        return workflow

        if high_lora and low_lora:
            low_node_id = "yunjii_animate_lora_low"
            workflow[low_node_id] = {
                "class_type": "WanVideoLoraSelect",
                "inputs": {
                    "lora": low_lora,
                    "strength": 1.0,
                    "low_mem_load": False,
                    "merge_loras": False,
                },
            }

            high_node_id = "yunjii_animate_lora_high"
            workflow[high_node_id] = {
                "class_type": "WanVideoLoraSelect",
                "inputs": {
                    "lora": high_lora,
                    "strength": 1.0,
                    "low_mem_load": False,
                    "merge_loras": False,
                    "prev_lora": [low_node_id, 0],
                },
            }

            inputs["lora"] = [high_node_id, 0]
            self._accel_lora_applied = True
            info("DirectAdapter", "注入Animate加速双LoRA(I2V-A14B蒸馏): high=%s + low=%s (步数→%d)", high_lora, low_lora, self.ANIMATE_ACCEL_STEPS)
        else:
            warn("DirectAdapter", "未找到Animate加速LoRA(high=%s, low=%s), 跳过注入 (步数≥%d, 无蒸馏加速)",
                 high_lora, low_lora, self.ANIMATE_MIN_STEPS)

        return workflow

    def _inject_enhance_lora(self, workflow: dict) -> dict:
        if not self._is_animate_model(workflow):
            return workflow

        import folder_paths
        available_loras = folder_paths.get_filename_list("loras")

        enhance_lora = None
        for avail in available_loras:
            if any(p in avail for p in self.ENHANCE_LORA_PATTERNS):
                enhance_lora = avail
                break

        if not enhance_lora:
            info("DirectAdapter", "未找到增强LoRA(%s), 跳过", self.ENHANCE_LORA_PATTERNS)
            return workflow

        model_node_id = None
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoModelLoader":
                model_node_id = node_id
                break
        if model_node_id is None:
            return workflow

        inputs = workflow[model_node_id].get("inputs", {})

        if "lora" not in inputs:
            return workflow

        lora_ref = inputs["lora"]
        if not isinstance(lora_ref, list) or len(lora_ref) < 1:
            return workflow

        last_lora_node_id = str(lora_ref[0])
        last_lora_node = workflow.get(last_lora_node_id, {})
        if not isinstance(last_lora_node, dict):
            return workflow

        existing_lora = last_lora_node.get("inputs", {}).get("lora", "")
        if any(p in existing_lora for p in self.ENHANCE_LORA_PATTERNS):
            info("DirectAdapter", "已有增强LoRA: %s", existing_lora)
            return workflow

        enhance_node_id = "yunjii_enhance_lora"
        workflow[enhance_node_id] = {
            "class_type": "WanVideoLoraSelect",
            "inputs": {
                "lora": enhance_lora,
                "strength": self.ENHANCE_LORA_STRENGTH,
                "low_mem_load": False,
                "merge_loras": False,
                "prev_lora": [last_lora_node_id, 0],
            },
        }

        inputs["lora"] = [enhance_node_id, 0]
        info("DirectAdapter", "注入增强LoRA: %s (strength=%.1f)", enhance_lora, self.ENHANCE_LORA_STRENGTH)

        return workflow

    def _inject_accel_lora(self, workflow: dict, height: int = 480) -> dict:
        self._accel_lora_applied = False

        if self._is_animate_model(workflow):
            self._remove_i2v_lora(workflow)
            return self._inject_animate_accel_lora(workflow)

        import folder_paths
        available_loras = folder_paths.get_filename_list("loras")

        if height <= 480:
            preferred_pattern = self.ACCEL_LORA_480P
            fallback_pattern = self.ACCEL_LORA_GENERAL
            resolution_label = "480p"
        else:
            preferred_pattern = self.ACCEL_LORA_GENERAL
            fallback_pattern = self.ACCEL_LORA_480P
            resolution_label = "720p+"

        lora_name = None
        for avail in available_loras:
            if preferred_pattern in avail:
                lora_name = avail
                break
        if lora_name is None:
            for avail in available_loras:
                if fallback_pattern in avail:
                    lora_name = avail
                    break

        if lora_name is None:
            warn("DirectAdapter", "未找到加速LoRA, 跳过注入 (候选: %s, %s)", preferred_pattern, fallback_pattern)
            return workflow

        model_node_id = None
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoModelLoader":
                model_node_id = node_id
                break
        if model_node_id is None:
            return workflow

        inputs = workflow[model_node_id].get("inputs", {})

        if "lora" in inputs:
            lora_ref = inputs["lora"]
            if isinstance(lora_ref, list) and len(lora_ref) >= 1:
                lora_node_id = str(lora_ref[0])
                lora_node = workflow.get(lora_node_id, {})
                if isinstance(lora_node, dict) and lora_node.get("class_type") == "WanVideoLoraSelect":
                    existing_lora = lora_node.get("inputs", {}).get("lora", "")
                    is_accel = any(p in existing_lora for p in [self.ACCEL_LORA_480P, self.ACCEL_LORA_GENERAL])
                    if is_accel:
                        self._accel_lora_applied = True
                        if preferred_pattern in existing_lora:
                            info("DirectAdapter", "已有加速LoRA匹配分辨率(%s): %s", resolution_label, existing_lora)
                        else:
                            lora_node["inputs"]["lora"] = lora_name
                            info("DirectAdapter", "分辨率(%s)不匹配, 切换加速LoRA: %s → %s", resolution_label, existing_lora, lora_name)
                        return workflow
                    else:
                        info("DirectAdapter", "已有用户LoRA(%s), 跳过加速LoRA注入", existing_lora)
                        return workflow
            return workflow

        lora_node_id = "yunjii_accel_lora"
        workflow[lora_node_id] = {
            "class_type": "WanVideoLoraSelect",
            "inputs": {
                "lora": lora_name,
                "strength": 1.0,
                "low_mem_load": False,
                "merge_loras": False,
            },
        }
        inputs["lora"] = [lora_node_id, 0]
        self._accel_lora_applied = True
        info("DirectAdapter", "自动注入加速LoRA(%s): %s (步数→%d)", resolution_label, lora_name, self.ACCEL_LORA_STEPS)

        return workflow

    def _ensure_distill_config(self, workflow: dict):
        if self._is_animate_model(workflow):
            for node_id, node_data in workflow.items():
                if not isinstance(node_data, dict):
                    continue
                if node_data.get("class_type") == "WanVideoSampler":
                    sampler_inputs = node_data.get("inputs", {})
                    if self._accel_lora_applied:
                        if "steps" in sampler_inputs:
                            sampler_inputs["steps"] = self.ANIMATE_ACCEL_STEPS
                            info("DirectAdapter", "Animate蒸馏步数 → %d", self.ANIMATE_ACCEL_STEPS)
                        if "cfg" in sampler_inputs:
                            sampler_inputs["cfg"] = self.ANIMATE_ACCEL_CFG
                            info("DirectAdapter", "Animate蒸馏CFG → %.1f (蒸馏模型不需要CFG)", self.ANIMATE_ACCEL_CFG)
                        if "scheduler" in sampler_inputs:
                            sampler_inputs["scheduler"] = self.ANIMATE_ACCEL_SCHEDULER
                            info("DirectAdapter", "Animate蒸馏调度器 → %s", self.ANIMATE_ACCEL_SCHEDULER)
                        if "shift" in sampler_inputs:
                            sampler_inputs["shift"] = self.ANIMATE_ACCEL_SHIFT
                            info("DirectAdapter", "Animate蒸馏shift → %.1f", self.ANIMATE_ACCEL_SHIFT)
                    else:
                        if "steps" in sampler_inputs:
                            current_steps = sampler_inputs["steps"]
                            min_steps = self.ANIMATE_MIN_STEPS
                            if current_steps < min_steps:
                                sampler_inputs["steps"] = min_steps
                                info("DirectAdapter", "Animate模型步数保护: %d → %d", current_steps, min_steps)
                        if "scheduler" in sampler_inputs:
                            old_scheduler = sampler_inputs.get("scheduler", "unipc")
                            if old_scheduler == "flowmatch_distill":
                                sampler_inputs["scheduler"] = "unipc"
                                info("DirectAdapter", "Animate模型调度器: flowmatch_distill → unipc")
                    break
            return

        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoSampler":
                sampler_inputs = node_data.get("inputs", {})
                if "steps" in sampler_inputs:
                    old_steps = sampler_inputs["steps"]
                    sampler_inputs["steps"] = self.ACCEL_LORA_STEPS
                    info("DirectAdapter", "自动调整步数: %d → %d", old_steps, self.ACCEL_LORA_STEPS)
                old_scheduler = sampler_inputs.get("scheduler", "unipc")
                sampler_inputs["scheduler"] = "flowmatch_distill"
                info("DirectAdapter", "自动调整调度器: %s → flowmatch_distill", old_scheduler)
                break

    def _inject_block_swap(self, workflow: dict) -> dict:
        model_node_id = None
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "WanVideoModelLoader":
                model_node_id = node_id
                break
        if model_node_id is None:
            return workflow

        inputs = workflow[model_node_id].get("inputs", {})
        if inputs.get("block_swap_args") is not None:
            return workflow

        try:
            import torch
            total_vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        except Exception:
            total_vram_gb = 24.0

        if total_vram_gb >= 48:
            return workflow

        blocks_to_swap = 20
        if total_vram_gb <= 16:
            blocks_to_swap = 34
        elif total_vram_gb <= 24:
            blocks_to_swap = 28

        block_swap_args = {
            "blocks_to_swap": blocks_to_swap,
            "offload_img_emb": True,
            "offload_txt_emb": True,
            "use_non_blocking": True,
            "vace_blocks_to_swap": 0,
            "prefetch_blocks": 1,
            "block_swap_debug": False,
        }
        inputs["block_swap_args"] = block_swap_args
        info("DirectAdapter", "自动注入block_swap: blocks_to_swap=%d (VRAM=%.0fGB)", blocks_to_swap, total_vram_gb)
        return workflow

    def _inject_context_options(self, workflow: dict, num_frames: int) -> dict:
        if self._is_animate_model(workflow):
            info("DirectAdapter", "Animate模型不支持ContextOptions, 跳过")
            return workflow

        sampler_node_id = None
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            ct = node_data.get("class_type", "")
            if ct in ("WanVideoSampler", "WanVideoSamplerFromSettings",
                       "WanAnimatePlus SamplerFromSettings"):
                sampler_node_id = node_id
                break
        if sampler_node_id is None:
            return workflow

        sampler_inputs = workflow[sampler_node_id].get("inputs", {})
        if "context_options" in sampler_inputs and sampler_inputs["context_options"] is not None:
            return workflow

        context_frames = min(81, num_frames)
        if num_frames <= context_frames:
            return workflow

        context_node_id = "yunjii_context_options"
        workflow[context_node_id] = {
            "class_type": "WanVideoContextOptions",
            "inputs": {
                "context_schedule": "uniform_standard",
                "context_frames": context_frames,
                "context_stride": 4,
                "context_overlap": 32,
                "freenoise": True,
                "verbose": False,
                "fuse_method": "linear",
            },
        }
        sampler_inputs["context_options"] = [context_node_id, 0]
        info("DirectAdapter", "注入ContextOptions: frames=%d, stride=4, overlap=32 (总帧数=%d)", context_frames, num_frames)

        return workflow

    def modify_workflow_for_segment(self, workflow, node_map, seg, ref_image_path, pose_dir="", run_id="", user_ref_path="", prev_video_path=""):
        import folder_paths
        import shutil

        wf = json.loads(json.dumps(workflow))
        wf = self._fix_fp8_scaled_t5(wf)
        wf = self._inject_accel_lora(wf, height=seg.params.get("height", 480))
        wf = self._inject_enhance_lora(wf)
        wf = self._inject_block_swap(wf)
        wf = self._inject_context_options(wf, num_frames=seg.target_frames)

        if ref_image_path and node_map.ref_image:
            target_w = seg.params.get("width", 832)
            target_h = seg.params.get("height", 480)
            cropped_ref = self._crop_ref_to_target_ratio(ref_image_path, target_w, target_h)
            img_name = self._copy_to_input(cropped_ref)
            info("DirectAdapter", "参考图注入: ref_image_path=%s, img_name=%s, node_map.ref_image=%s, node_in_wf=%s",
                 ref_image_path, img_name, node_map.ref_image, node_map.ref_image in wf)
            if img_name and node_map.ref_image in wf:
                old_image = wf[node_map.ref_image].get("inputs", {}).get("image", "")
                wf[node_map.ref_image]["inputs"]["image"] = img_name
                info("DirectAdapter", "参考图已更新: %s → %s", old_image, img_name)
                verify_image = wf[node_map.ref_image]["inputs"].get("image", "")
                info("DirectAdapter", "参考图验证: LoadImage节点[%s] image='%s', 节点完整inputs=%s",
                     node_map.ref_image, verify_image, json.dumps(wf[node_map.ref_image]["inputs"], ensure_ascii=False))
            else:
                info("DirectAdapter", "参考图注入失败: img_name='%s', node_in_wf=%s",
                     img_name, node_map.ref_image in wf if node_map.ref_image else "N/A")
        else:
            info("DirectAdapter", "参考图跳过: ref_image_path=%s, node_map.ref_image=%s",
                 ref_image_path, node_map.ref_image)

        if node_map.animate_embeds and node_map.animate_embeds in wf:
            ae = wf[node_map.animate_embeds]
            ae["inputs"]["width"] = seg.params.get("width", 832)
            ae["inputs"]["height"] = seg.params.get("height", 480)
            aligned_frames = max(9, ((seg.target_frames - 1) // 4) * 4 + 1)
            ae["inputs"]["num_frames"] = aligned_frames
            if "frame_window_size" in ae.get("inputs", {}):
                ae["inputs"]["frame_window_size"] = aligned_frames
            info("DirectAdapter", "AnimateEmbeds num_frames: %d → %d (4k+1对齐), frame_window_size=%d (禁用looping)",
                 seg.target_frames, aligned_frames, aligned_frames)

            if pose_dir and "pose_images" not in ae.get("inputs", {}):
                if os.path.isdir(pose_dir):
                    pose_node_id = "yunjii_pose_loader"
                    wf[pose_node_id] = {
                        "class_type": "YunjiiLoadPoseImages",
                        "inputs": {
                            "目录路径": pose_dir,
                            "目标帧数": aligned_frames,
                        },
                    }
                    ae["inputs"]["pose_images"] = [pose_node_id, 0]
                    ae["inputs"]["pose_strength"] = self.ANIMATE_POSE_STRENGTH
                    ae["inputs"]["face_strength"] = self.ANIMATE_FACE_STRENGTH
                    info("DirectAdapter", "注入姿态引导: %s (%d帧), pose_strength=%.1f, face_strength=%.1f (直接加载PNG)",
                         pose_dir, aligned_frames, self.ANIMATE_POSE_STRENGTH, self.ANIMATE_FACE_STRENGTH)
                else:
                    warn("DirectAdapter", "姿态目录不存在: %s", pose_dir)

            if "face_images" not in ae.get("inputs", {}) or ae["inputs"].get("face_images") is None:
                if ref_image_path and node_map.ref_image and node_map.ref_image in wf:
                    ae["inputs"]["face_images"] = [node_map.ref_image, 0]
                    info("DirectAdapter", "注入face_images: 使用参考图作为面部参考 (face_strength=%.1f)", self.ANIMATE_FACE_STRENGTH)

        if node_map.sampler and node_map.sampler in wf:
            sampler = wf[node_map.sampler]
            if "steps" in sampler.get("inputs", {}):
                sampler["inputs"]["steps"] = seg.params.get("steps", 30)
            if "cfg" in sampler.get("inputs", {}):
                sampler["inputs"]["cfg"] = seg.params.get("cfg", 6.0)

        if node_map.text_encode and node_map.text_encode in wf:
            te = wf[node_map.text_encode]
            if "positive_prompt" in te.get("inputs", {}):
                old_prompt = te["inputs"]["positive_prompt"][:50]
                te["inputs"]["positive_prompt"] = seg.prompt
                info("DirectAdapter", "提示词已更新(positive_prompt): %s... → %s...", old_prompt, seg.prompt[:50])
            elif "prompt" in te.get("inputs", {}):
                old_prompt = te["inputs"]["prompt"][:50]
                te["inputs"]["prompt"] = seg.prompt
                info("DirectAdapter", "提示词已更新(prompt): %s... → %s...", old_prompt, seg.prompt[:50])
            if "negative_prompt" in te.get("inputs", {}) and seg.negative_prompt:
                te["inputs"]["negative_prompt"] = seg.negative_prompt

        if node_map.video_combine and node_map.video_combine in wf:
            vc = wf[node_map.video_combine]
            vc_inputs = vc.get("inputs", {})
            if "filename_prefix" in vc_inputs:
                sub_dir = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
                vc_inputs["filename_prefix"] = f"yunjii_v2v/{sub_dir}/segments"

        if self._accel_lora_applied:
            self._ensure_distill_config(wf)
        elif self._is_animate_model(wf):
            self._ensure_distill_config(wf)

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
                info("DirectAdapter", "_copy_to_input: 复制 %s → %s", src_path, dst)
            else:
                info("DirectAdapter", "_copy_to_input: 文件已存在, 跳过复制: %s (size=%d)", dst, os.path.getsize(dst))
            return fname
        except Exception as e:
            info("DirectAdapter", "_copy_to_input 异常: %s", e)
            return ""

    def _crop_ref_to_target_ratio(self, ref_image_path: str, target_w: int, target_h: int) -> str:
        try:
            import cv2
            img = cv2.imread(ref_image_path)
            if img is None:
                return ref_image_path

            img_h, img_w = img.shape[:2]
            target_ratio = target_w / target_h
            img_ratio = img_w / img_h

            if abs(img_ratio - target_ratio) < 0.05:
                return ref_image_path

            if img_ratio > target_ratio:
                new_w = int(img_h * target_ratio)
                x_start = (img_w - new_w) // 2
                cropped = img[:, x_start:x_start + new_w]
            else:
                new_h = int(img_w / target_ratio)
                y_start = (img_h - new_h) // 2
                cropped = img[y_start:y_start + new_h, :]

            out_name = f"yunjii_cropped_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(os.path.dirname(ref_image_path), out_name)
            cv2.imwrite(out_path, cropped)

            info("DirectAdapter", "参考图裁剪: %dx%d(ratio=%.2f) → %dx%d(ratio=%.2f), 目标ratio=%.2f",
                 img_w, img_h, img_ratio, cropped.shape[1], cropped.shape[0],
                 cropped.shape[1] / cropped.shape[0], target_ratio)

            return out_path
        except Exception as e:
            info("DirectAdapter", "参考图裁剪异常: %s, 使用原图", e)
            return ref_image_path
