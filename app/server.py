"""零依赖的本地 HTTP 服务。

为什么不用 FastAPI：这个 demo 要给一个没装过任何 Python 包的机器跑。
标准库够用（几个 JSON 接口 + 静态文件），少一层依赖就少一个"跑不起来"的理由。
FastAPI 在别的项目里用得多，这里刻意不用，是因为这里的首要目标是"对方打开就能看"。

音频走 raw body（Content-Type: audio/wav），不走 multipart——
标准库解析 multipart 要手写边界处理，徒增出错面；raw body 简单且够用。
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import asr, redact as redactor  # noqa: E402

WEB_DIR = ROOT / "web"
MAX_AUDIO = 40 * 1024 * 1024


def _json(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _split_words(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [w.strip() for w in unquote(raw).replace("，", ",").split(",") if w.strip()]


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceRedactDemo/1.0"

    # 默认的日志会把 URL 打出来。URL 里可能带词表（人名），所以只打路径不打 query。
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # ── 路由 ──────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return _json(self, self._status())
        if path in ("/", "/index.html"):
            return self._serve_file(WEB_DIR / "index.html")
        # 静态资源，禁止越界
        target = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR.resolve() in target.parents and target.is_file():
            return self._serve_file(target)
        self.send_error(404, "not found")

    def do_POST(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        try:
            if path == "/api/redact":
                return self._handle_redact()
            if path == "/api/transcribe":
                return self._handle_transcribe(qs)
            if path == "/api/pipeline":
                return self._handle_pipeline(qs)
            self.send_error(404, "not found")
        except asr.ASRError as e:
            _json(self, {"error": str(e)}, 400)
        except Exception:
            traceback.print_exc()
            _json(self, {"error": "服务内部错误，详情见终端日志"}, 500)

    # ── 具体处理 ──────────────────────────────────────────────────────────
    def _status(self) -> dict:
        return {
            "asr": {
                "available": asr.available(),
                "model": str(asr._find_model() or ""),
                "hint": "" if asr.available() else "未检测到 faster-whisper 或本地模型，语音功能关闭；文字脱敏仍可用",
            },
            "pinyin": _has("pypinyin"),
            "opencc": _has("opencc"),
            "limits": {"max_audio_mb": MAX_AUDIO // 1024 // 1024},
        }

    def _handle_redact(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 4 * 1024 * 1024:
            return _json(self, {"error": "文本过长"}, 413)
        data = json.loads(self.rfile.read(n) or b"{}")
        result = redactor.redact(
            data.get("text", ""),
            persons=_split_words(data.get("persons", "")) if isinstance(data.get("persons"), str) else data.get("persons") or [],
            orgs=_split_words(data.get("orgs", "")) if isinstance(data.get("orgs"), str) else data.get("orgs") or [],
            use_name_heuristic=bool(data.get("heuristic", True)),
        )
        _json(self, result.to_dict())

    def _read_audio(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise asr.ASRError("没有收到音频数据")
        if n > MAX_AUDIO:
            raise asr.ASRError("音频超过 40MB")
        return self.rfile.read(n)

    def _handle_transcribe(self, qs):
        data = self._read_audio()
        hotwords = _split_words((qs.get("hotwords") or [""])[0])
        t = asr.transcribe(data, hotwords=hotwords)
        _json(self, {"transcript": t.to_dict()})

    def _handle_pipeline(self, qs):
        data = self._read_audio()
        hotwords = _split_words((qs.get("hotwords") or [""])[0])
        persons = _split_words((qs.get("persons") or [""])[0])
        t = asr.transcribe(data, hotwords=hotwords or persons)
        result = redactor.redact(t.text, persons=persons or hotwords)
        _json(self, {"transcript": t.to_dict(), "redaction": result.to_dict()})

    # ── 静态文件 ──────────────────────────────────────────────────────────
    def _serve_file(self, path: Path):
        if not path.is_file():
            return self.send_error(404, "not found")
        raw = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def serve(host: str = "127.0.0.1", port: int = 8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"  → http://{host}:{port}")
    print(f"  → ASR {'已就绪' if asr.available() else '不可用（仅文字脱敏）'}")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


if __name__ == "__main__":
    port = int(os.getenv("VRD_PORT", "8765"))
    srv = serve(port=port)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n停止")
