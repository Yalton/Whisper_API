"""Lazy, idle-evictable ASR backends.

The HTTP layer consumes one normalized result shape, while model-specific
details live here. Whisper remains the default; Qwen3-ASR can be selected with
ASR_BACKEND=qwen3.
"""

from .config import settings
from .models import TranscriptionResult, TranscriptionSegment

import aiofiles
import asyncio
import contextlib
import gc
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from typing import Any


_model = None
_active = 0
_last_used = 0.0
_lock = threading.Lock()


def now() -> float:
    return time.monotonic()


def model_is_loaded() -> bool:
    with _lock:
        return _model is not None


def mark_used(at: float | None = None) -> None:
    global _last_used
    with _lock:
        _last_used = now() if at is None else at


def _load_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.MODEL_SIZE,
        device=settings.COMPUTE_DEVICE,
        compute_type="float16" if settings.COMPUTE_DEVICE == "cuda" else "int8",
    )


def _qwen_dtype(torch):
    configured = settings.QWEN_DTYPE
    if configured != "auto":
        try:
            return getattr(torch, configured)
        except AttributeError as exc:
            raise ValueError(f"Unsupported QWEN_DTYPE={configured!r}") from exc
    if settings.COMPUTE_DEVICE == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _load_qwen_model():
    import torch
    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if settings.COMPUTE_DEVICE == "cuda" else "cpu"
    common = {
        "dtype": _qwen_dtype(torch),
        "device_map": device,
    }
    if settings.QWEN_ATTN_IMPLEMENTATION:
        common["attn_implementation"] = settings.QWEN_ATTN_IMPLEMENTATION

    kwargs = {
        **common,
        "max_inference_batch_size": settings.QWEN_MAX_BATCH_SIZE,
        "max_new_tokens": settings.QWEN_MAX_NEW_TOKENS,
    }
    if settings.QWEN_ALIGNER_MODEL:
        kwargs["forced_aligner"] = settings.QWEN_ALIGNER_MODEL
        kwargs["forced_aligner_kwargs"] = dict(common)
    return Qwen3ASRModel.from_pretrained(settings.QWEN_MODEL, **kwargs)


def _load_model():
    if settings.ASR_BACKEND == "whisper":
        return _load_whisper_model()
    if settings.ASR_BACKEND == "qwen3":
        return _load_qwen_model()
    raise ValueError(
        f"Unsupported ASR_BACKEND={settings.ASR_BACKEND!r}; use 'whisper' or 'qwen3'"
    )


def reset_model_state() -> None:
    """Drop the model and counters. Also used as a test hook."""
    global _model, _active, _last_used
    with _lock:
        _model = None
        _active = 0
        _last_used = 0.0
    _release_runtime_memory()


def _release_runtime_memory() -> None:
    gc.collect()
    if settings.ASR_BACKEND == "qwen3":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


@contextlib.contextmanager
def model_session():
    global _model, _active, _last_used
    with _lock:
        if _model is None:
            start = datetime.now()
            settings.logger.info(
                "Initializing ASR model backend=%s model=%s device=%s",
                settings.ASR_BACKEND,
                configured_model_name(),
                settings.COMPUTE_DEVICE,
            )
            _model = _load_model()
            settings.logger.info(
                "ASR model initialized in %.2f seconds",
                (datetime.now() - start).total_seconds(),
            )
        _active += 1
        model = _model
    try:
        yield model
    finally:
        with _lock:
            _active -= 1
            _last_used = now()


def maybe_evict_idle_model(idle_timeout: float, now: float | None = None) -> bool:
    global _model
    with _lock:
        if _model is None or _active > 0:
            return False
        current = time.monotonic() if now is None else now
        idle_for = current - _last_used
        if idle_for < idle_timeout:
            return False
        _model = None
    _release_runtime_memory()
    settings.logger.info(
        "Released idle %s model after %.0fs idle", settings.ASR_BACKEND, idle_for
    )
    return True


async def idle_eviction_loop(
    idle_timeout: float | None = None, interval: float | None = None
) -> None:
    timeout = settings.IDLE_TIMEOUT_SECONDS if idle_timeout is None else idle_timeout
    if timeout <= 0:
        settings.logger.info("ASR idle eviction disabled")
        return
    poll = max(5.0, min(timeout / 4.0, 60.0)) if interval is None else interval
    while True:
        await asyncio.sleep(poll)
        try:
            await asyncio.to_thread(maybe_evict_idle_model, timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            settings.logger.exception("Idle eviction check failed")


os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)


def purge_upload_directory() -> int:
    """Remove everything left in the upload directory. Run at startup only.

    Requests delete their upload in a finally block, but a process that is
    SIGKILLed (OOM, docker kill) never reaches it, so files and chunk
    directories from the killed request would otherwise sit in the container's
    writable layer until it is recreated.
    """
    removed = 0
    try:
        entries = list(os.scandir(settings.UPLOAD_DIRECTORY))
    except FileNotFoundError:
        return 0
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
            removed += 1
        except OSError:
            settings.logger.warning("Could not remove stale upload %s", entry.path)
    if removed:
        settings.logger.info(
            "Removed %d stale entries from %s", removed, settings.UPLOAD_DIRECTORY
        )
    return removed


async def save_uploaded_file(upload_file, path: str):
    async with aiofiles.open(path, "wb") as out_file:
        while chunk := await upload_file.read(1024 * 1024):
            await out_file.write(chunk)
    settings.logger.info("Saved file to %s", path)


def configured_model_name() -> str:
    return settings.QWEN_MODEL if settings.ASR_BACKEND == "qwen3" else settings.MODEL_SIZE


def _audio_duration(file_path: str) -> float:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            file_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(process.stdout)["format"]["duration"])


def _needs_chunking(duration: float) -> bool:
    return settings.CHUNK_SECONDS > 0 and duration > settings.CHUNK_SECONDS


def _ffmpeg_to_mono_16k(file_path: str, output_args: list[str]) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            file_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            *output_args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _split_audio(file_path: str, chunk_seconds: float, out_dir: str) -> list[tuple[float, str]]:
    """Split audio into 16 kHz mono WAV chunks; return (offset, path) pairs.

    Offsets are the running sum of each chunk's measured duration, so they stay
    exact even if the segment muxer does not cut at exactly chunk_seconds.
    """
    _ffmpeg_to_mono_16k(
        file_path,
        [
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            os.path.join(out_dir, "chunk_%05d.wav"),
        ],
    )
    chunks = []
    offset = 0.0
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        chunks.append((offset, path))
        offset += _audio_duration(path)
    return chunks


def _shift_segment(segment: TranscriptionSegment, offset: float) -> TranscriptionSegment:
    return TranscriptionSegment(
        start=segment.start + offset,
        end=segment.end + offset,
        text=segment.text,
        words=[
            {
                **word,
                "start": word["start"] + offset,
                "end": word["end"] + offset,
            }
            for word in segment.words
        ]
        if segment.words is not None
        else None,
    )


def _transcribe_chunked(transcribe_one, file_path: str, duration: float) -> TranscriptionResult:
    """Transcribe chunk by chunk and merge into one result.

    Language and its probability come from the first chunk; the reported
    duration is that of the original file.
    """
    chunk_dir = tempfile.mkdtemp(prefix="chunks_", dir=settings.UPLOAD_DIRECTORY)
    try:
        chunks = _split_audio(file_path, settings.CHUNK_SECONDS, chunk_dir)
        settings.logger.info(
            "Transcribing %.0fs of audio in %d chunks of up to %.0fs",
            duration,
            len(chunks),
            settings.CHUNK_SECONDS,
        )
        segments = []
        first = None
        for index, (offset, chunk_path) in enumerate(chunks):
            settings.logger.info(
                "Transcribing chunk %d/%d at offset %.0fs", index + 1, len(chunks), offset
            )
            result = transcribe_one(chunk_path)
            if first is None:
                first = result
            segments.extend(_shift_segment(segment, offset) for segment in result.segments)
            # Drop the finished chunk now rather than holding all of them on disk.
            os.remove(chunk_path)
        if first is None:
            raise ValueError(f"ffmpeg produced no chunks for {file_path}")
        return TranscriptionResult(
            segments=segments,
            language=first.language,
            language_probability=first.language_probability,
            duration=duration,
        )
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


_QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}
_QWEN_LANGUAGE_CODES = {name.lower(): code for code, name in _QWEN_LANGUAGE_NAMES.items()}


def _qwen_language(language: str | None) -> str | None:
    if not language:
        return None
    return _QWEN_LANGUAGE_NAMES.get(language.lower(), language)


def _language_code(language: str) -> str:
    # Mixed-language results such as "Chinese,English" have no single ISO code.
    return _QWEN_LANGUAGE_CODES.get(language.lower(), language)


def _transcribe_whisper(model, file_path, include_word_timestamps, language):
    options = dict(
        beam_size=5,
        word_timestamps=include_word_timestamps,
        language=language,
    )
    if settings.WHISPER_BATCH_SIZE > 0:
        # The sequential decoder runs one 30 s window at a time and leaves the
        # GPU mostly idle; batching windows is several times faster on long audio.
        from faster_whisper import BatchedInferencePipeline

        segments, info = BatchedInferencePipeline(model=model).transcribe(
            file_path, batch_size=settings.WHISPER_BATCH_SIZE, **options
        )
    else:
        segments, info = model.transcribe(file_path, **options)
    normalized = [
        TranscriptionSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            words=[
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in segment.words
            ]
            if include_word_timestamps
            else None,
        )
        for segment in segments
    ]
    return TranscriptionResult(
        segments=normalized,
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
    )


def _transcribe_qwen(model, file_path, include_word_timestamps, language):
    if include_word_timestamps and not settings.QWEN_ALIGNER_MODEL:
        raise ValueError(
            "Word timestamps require QWEN_ALIGNER_MODEL to be configured"
        )
    result = model.transcribe(
        audio=file_path,
        language=_qwen_language(language),
        return_time_stamps=include_word_timestamps,
    )[0]
    duration = _audio_duration(file_path)
    words = None
    start, end = 0.0, duration
    if include_word_timestamps:
        words = [
            {
                "start": item.start_time,
                "end": item.end_time,
                "word": item.text,
                "probability": None,
            }
            for item in result.time_stamps
        ]
        if words:
            start, end = words[0]["start"], words[-1]["end"]
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=start, end=end, text=result.text, words=words
            )
        ],
        language=_language_code(result.language),
        language_probability=None,
        duration=duration,
    )


async def detect_language(file_path: str):
    if settings.ASR_BACKEND == "whisper":
        def run_whisper_detection():
            # transcribe() decodes the whole file before detecting, so for
            # long audio detect on the first chunk only.
            if not _needs_chunking(_audio_duration(file_path)):
                with model_session() as model:
                    _, info = model.transcribe(file_path)
                    return info.language, info.language_probability
            chunk_dir = tempfile.mkdtemp(prefix="detect_", dir=settings.UPLOAD_DIRECTORY)
            try:
                head = os.path.join(chunk_dir, "head.wav")
                _ffmpeg_to_mono_16k(file_path, ["-t", str(settings.CHUNK_SECONDS), head])
                with model_session() as model:
                    _, info = model.transcribe(head)
                    return info.language, info.language_probability
            finally:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        return await asyncio.to_thread(run_whisper_detection)

    result = await transcribe_audio(file_path)
    return result.language, result.language_probability


async def transcribe_audio(
    file_path: str,
    include_word_timestamps: bool = False,
    language: str | None = None,
) -> TranscriptionResult:
    def transcribe_one(path):
        with model_session() as model:
            if settings.ASR_BACKEND == "qwen3":
                return _transcribe_qwen(
                    model, path, include_word_timestamps, language
                )
            return _transcribe_whisper(
                model, path, include_word_timestamps, language
            )

    def run_transcription():
        duration = _audio_duration(file_path)
        if _needs_chunking(duration):
            return _transcribe_chunked(transcribe_one, file_path, duration)
        return transcribe_one(file_path)

    return await asyncio.to_thread(run_transcription)
