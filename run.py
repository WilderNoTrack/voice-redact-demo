"""一键启动：起服务 + 打开浏览器。

    python run.py            # 默认 127.0.0.1:8765
    python run.py --port 9000
    python run.py --no-open  # 不自动开浏览器
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser

from app import asr, server


def main() -> None:
    ap = argparse.ArgumentParser(description="语音转写与文本脱敏 demo")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n  语音转写与文本脱敏 · 本地运行")
    print("  " + "-" * 46)
    httpd = server.serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"  页面  {url}")
    print(f"  健康检查  {url}/api/status")
    if not asr.available():
        print("\n  提示：未检测到 faster-whisper 或本地模型，语音功能关闭。")
        print("        文字脱敏不受影响，可直接粘贴文本测试。")
        print("        启用语音：pip install faster-whisper，")
        print("        并把 VRD_MODEL_DIR 指向 faster-whisper 的 ct2 模型目录。")
    print("\n  Ctrl+C 停止\n")

    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  已停止")
        httpd.shutdown()


if __name__ == "__main__":
    main()
