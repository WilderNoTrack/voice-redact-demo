"""本地语音转写。

这条链路的三个约束，决定了代码的样子：

1. 音频不出本机。录音以 bytes 进、以文本出，全程在内存里，
   不写临时文件、不进日志。产品上这是硬边界，不是优化项。
2. 模型可能拖死服务。whisper 在小机器上加载要几秒到几十秒，推理时还可能吃满内存，
   所以模型加载和推理都放在带锁的单例里，失败要能降级到"请改用文字输入"，
   而不是让整个服务挂掉。
3. 小模型认不出专有名词。whisper-small 对景点名、人名、术语基本靠猜，
   所以有两个补救：解码前用 initial_prompt 注入热词（改先验），
   解码后用词表 + 拼音做同音纠错（改结果）。前者便宜，后者可控，两个都要有。
"""

from __future__ import annotations

import difflib
import io
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

_MODEL = None
_BACKEND = ""
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()

# 默认找 HF 缓存里的 whisper-small ct2；也可以用环境变量指定
DEFAULT_MODEL_DIRS = [
    Path.home() / ".cache/huggingface/hub/models--whisper-small-ct2",
    Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots",
]


class ASRError(RuntimeError):
    """给用户看的话，不要抛原始异常堆栈。"""


@dataclass
class Transcript:
    text: str
    segments: list[dict] = field(default_factory=list)
    duration: float = 0.0          # 音频时长
    elapsed: float = 0.0           # 推理耗时
    corrections: list[dict] = field(default_factory=list)   # 词表纠错记录
    backend: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": self.segments,
            "duration": round(self.duration, 2),
            "elapsed": round(self.elapsed, 2),
            "rtf": round(self.elapsed / self.duration, 3) if self.duration else None,
            "corrections": self.corrections,
            "backend": self.backend,
        }


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return _find_model() is not None


def _find_model() -> Path | None:
    env = os.getenv("VRD_MODEL_DIR")
    if env and Path(env).exists():
        return Path(env)
    for p in DEFAULT_MODEL_DIRS:
        if p.exists():
            if p.name == "snapshots":       # HF 缓存结构，取第一个快照
                subs = sorted([d for d in p.iterdir() if d.is_dir()])
                if subs:
                    return subs[0]
            elif (p / "model.bin").exists() or (p / "config.json").exists():
                return p
    return None


def load_model():
    """懒加载 + 双检锁，GPU 不可用时自动降级 CPU。

    这台机器上 ctranslate2 能检测到 CUDA，但 cuBLAS 的 DLL 不一定在 PATH 里，
    真正的失败发生在构造 WhisperModel 那一刻（`cublas64_12.dll is not found`）。
    让用户去修 DLL 路径等于把工程债推给使用者，所以这里直接退回 CPU + int8：
    慢一点，但一定能跑。想用 GPU 就设 VRD_CUBLAS_DIR 指向自带的 cuBLAS 目录。
    """
    global _MODEL, _BACKEND
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = _find_model()
        if path is None:
            raise ASRError("没有找到本地模型目录，请设置 VRD_MODEL_DIR 指向 faster-whisper 的 ct2 模型")
        try:
            from faster_whisper import WhisperModel
            import ctranslate2
        except ImportError as e:
            raise ASRError(f"缺少 faster-whisper 依赖：{e}") from e

        cublas = os.getenv("VRD_CUBLAS_DIR")
        if cublas and Path(cublas).is_dir() and os.name == "nt":
            os.environ["PATH"] = cublas + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(cublas)
            except (AttributeError, OSError):
                pass

        want = os.getenv("VRD_DEVICE")
        attempts: list[tuple[str, str]] = []
        if want:
            attempts.append((want, "float16" if want == "cuda" else "int8"))
        else:
            if ctranslate2.get_cuda_device_count():
                attempts.append(("cuda", "float16"))
            attempts.append(("cpu", "int8"))

        last_err = None
        for device, compute in attempts:
            try:
                _MODEL = WhisperModel(
                    str(path), device=device, compute_type=compute,
                    local_files_only=True,
                    cpu_threads=int(os.getenv("VRD_THREADS", "4")),
                )
                _BACKEND = f"{device}/{compute}"
                return _MODEL
            except Exception as e:      # noqa: BLE001 - 任何加载失败都应能降级
                last_err = e
                continue
        raise ASRError(f"模型加载失败：{last_err}") from last_err


def backend() -> str:
    return _BACKEND


def _switch_to_cpu():
    """推理期发现 GPU 不可用，就地换成 CPU 模型。"""
    global _MODEL, _BACKEND
    from faster_whisper import WhisperModel
    path = _find_model()
    if path is None:
        raise ASRError("没有找到本地模型目录")
    _MODEL = WhisperModel(
        str(path), device="cpu", compute_type="int8", local_files_only=True,
        cpu_threads=int(os.getenv("VRD_THREADS", "4")),
    )
    _BACKEND = "cpu/int8（GPU 不可用，已降级）"
    return _MODEL


def decode_wav(data: bytes):
    """自己解 WAV，不交给 PyAV。

    两个原因：一是 faster-whisper 内部走 av.open 解码，PyAV 版本一换就报
    `metadata_errors` 之类的参数错，这种依赖不该让用户去修；二是我们本来
    就只接受 16kHz/单声道/16bit，格式单一，用标准库 wave 解既快又不会出事。
    解出来直接喂 float32 数组给模型，音频始终在内存里。
    """
    import numpy as np
    try:
        with wave.open(io.BytesIO(data), "rb") as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
                raise ASRError(
                    f"需要 16kHz / 单声道 / 16bit 的 WAV，当前是 "
                    f"{f.getframerate()}Hz / {f.getnchannels()} 声道 / {f.getsampwidth() * 8}bit"
                )
            n = f.getnframes()
            raw = f.readframes(n)
    except wave.Error as e:
        raise ASRError(f"音频格式不对：{e}") from e
    if len(raw) != n * 2:
        raise ASRError("音频数据不完整，可能上传中断")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, n / 16000


def transcribe(data: bytes, hotwords: list[str] | None = None, language: str = "zh") -> Transcript:
    samples, dur = decode_wav(data)
    if dur < 0.3:
        raise ASRError("录音太短，请重录")

    import numpy as np
    if float(np.sqrt(np.mean(samples * samples))) < 0.003:
        raise ASRError("没有听到清楚的人声，请靠近麦克风或换一段音频")

    model = load_model()
    prompt = "，".join(hotwords) if hotwords else None

    with _INFER_LOCK:                    # 单卡单模型，串行更稳
        t0 = time.perf_counter()
        for attempt in range(2):
            try:
                segments, _info = model.transcribe(
                    samples,
                    language=language,
                    beam_size=3,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=prompt,
                )
                # segments 是生成器，异常要到迭代时才抛，所以必须把列表化放进 try
                segs = [
                    {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
                    for s in segments if s.no_speech_prob < 0.7
                ]
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                gpu_issue = any(k in msg for k in ("cublas", "cudnn", "dll", "cuda"))
                if attempt == 0 and gpu_issue and _BACKEND.startswith("cuda"):
                    # cuBLAS 确实是在推理时才加载，构造阶段不报错，
                    # 所以降级只能放到这里：换 CPU 重试一次，而不是把错误抛给用户。
                    model = _switch_to_cpu()
                    continue
                raise ASRError(f"识别失败：{e}") from e
        else:
            raise ASRError("识别失败：已尝试 GPU 与 CPU，均未成功")
        elapsed = time.perf_counter() - t0

    raw = "".join(s["text"] for s in segs).strip()
    if not raw:
        raise ASRError("这段没有识别出内容，请重录或改用文字输入")

    text = to_simplified(raw)
    text, corrections = correct_with_lexicon(text, hotwords or [])

    # 修正记录要能被界面标出来，否则用户不知道哪里被改过
    return Transcript(
        text=text, segments=segs, duration=dur, elapsed=elapsed,
        corrections=corrections, backend=_BACKEND,
    )


def to_simplified(text: str) -> str:
    try:
        from opencc import OpenCC
    except ImportError:
        return text
    return OpenCC("t2s").convert(text)


# ── 词表纠错 ──────────────────────────────────────────────────────────────

def _pinyin(s: str) -> list[str] | None:
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return None
    return lazy_pinyin(s)


def _similar(a: str, b: str) -> float:
    """优先比拼音，没有 pypinyin 就退回字符级。

    比拼音是关键：我们要抓的是"同音不同字"（张山 / 张珊），
    字符级相似度对这种错误几乎无感。
    """
    pa, pb = _pinyin(a), _pinyin(b)
    if pa and pb:
        return difflib.SequenceMatcher(None, pa, pb, autojunk=False).ratio()
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def correct_with_lexicon(text: str, lexicon: list[str], threshold: float = 0.88) -> tuple[str, list[dict]]:
    """在文本里找"听起来像词表词、但写错了"的片段并替换。

    只在拼音几乎一致（≥ threshold）且字面不同的时候动手，
    宁可漏改，不要错改——错改会把人名改成另一个人。
    """
    if not lexicon or not text:
        return text, []

    changes: list[dict] = []
    for word in sorted({w for w in lexicon if len(w) >= 2}, key=len, reverse=True):
        n = len(word)
        i = 0
        while i <= len(text) - n:
            cand = text[i:i + n]
            if cand == word:
                i += n
                continue
            # 只处理全中文的候选，避免把数字、标点卷进来
            if not all("\u4e00" <= c <= "\u9fa5" for c in cand):
                i += 1
                continue
            score = _similar(cand, word)
            if score >= threshold:
                text = text[:i] + word + text[i + n:]
                changes.append({"from": cand, "to": word, "score": round(score, 3)})
                i += n
            else:
                i += 1
    return text, changes
