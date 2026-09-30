"""Windows 控制台中文输出兼容：把 stdout/stderr 切成 UTF-8。"""
import sys


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:      # noqa: BLE001
                pass
