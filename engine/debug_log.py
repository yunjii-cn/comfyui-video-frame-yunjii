import os
import sys
import time
import traceback
import threading
import logging
from datetime import datetime

_logger = logging.getLogger("yunjii")
_logger.setLevel(logging.DEBUG)

_initialized = False
_log_dir = ""
_current_log_file = ""
_session_id = ""


def init_log_dir():
    global _log_dir, _initialized
    if _initialized:
        return _log_dir

    try:
        import folder_paths
        _log_dir = os.path.join(folder_paths.get_output_directory(), "yunjii_logs")
    except Exception:
        _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

    os.makedirs(_log_dir, exist_ok=True)
    _initialized = True
    return _log_dir


def get_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_log_file():
    global _current_log_file, _session_id
    if _current_log_file and os.path.isfile(_current_log_file):
        return _current_log_file

    log_dir = init_log_dir()
    _session_id = get_session_id()
    _current_log_file = os.path.join(log_dir, f"yunjii_{_session_id}.log")
    return _current_log_file


def _write(level, tag, msg, *args):
    log_file = get_log_file()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    content = msg % args if args else msg

    line = f"[{now}] [{level}] [{tag}] {content}\n"

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

    getattr(_logger, level.lower(), _logger.info)(f"[{tag}] {content}")


def debug(tag, msg, *args):
    _write("DEBUG", tag, msg, *args)


def info(tag, msg, *args):
    _write("INFO", tag, msg, *args)


def warn(tag, msg, *args):
    _write("WARNING", tag, msg, *args)


def error(tag, msg, *args):
    _write("ERROR", tag, msg, *args)


def exception(tag, msg, *args):
    content = msg % args if args else msg
    tb = traceback.format_exc()
    _write("ERROR", tag, f"{content}\n{tb}")


def node_start(node_name, **params):
    info(node_name, "===== 节点开始执行 =====")
    for k, v in params.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        debug(node_name, "  参数 %s = %s", k, val_str)


def node_end(node_name, results_summary=""):
    info(node_name, "===== 节点执行完成 ===== %s", results_summary)


def node_error(node_name, error_msg):
    error(node_name, "===== 节点执行失败 ===== %s", error_msg)


def data_flow(from_node, from_output, to_node, to_input, value_type, value_preview=""):
    debug("DATA_FLOW", "%s.%s ──→ %s.%s [%s] %s",
          from_node, from_output, to_node, to_input, value_type, value_preview)


def separator(title=""):
    log_file = get_log_file()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 60
    line = f"\n{sep}\n[{now}] {title}\n{sep}\n" if title else f"\n{sep}\n"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def get_log_dir_path():
    return init_log_dir()


def get_current_log_path():
    return get_log_file()


def list_recent_logs(count=10):
    log_dir = init_log_dir()
    if not os.path.isdir(log_dir):
        return []
    files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("yunjii_") and f.endswith(".log")],
        reverse=True,
    )
    return files[:count]


def read_log(filename, tail_lines=100):
    log_dir = init_log_dir()
    path = os.path.join(log_dir, filename)
    if not os.path.isfile(path):
        return f"⚠ 日志文件不存在: {filename}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-tail_lines:])
    except Exception as e:
        return f"⚠ 读取日志失败: {e}"
