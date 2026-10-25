"""Lazy, idle-evictable ASR backends.

The HTTP layer consumes one normalized result shape, while model-specific
details live here. Whisper and NVIDIA Parakeet are both available at once:
each request picks one, and each model loads on first use and is released
after it has been idle, independently of the other.
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


BACKENDS = ("whisper", "parakeet")

_models: dict[str, Any] = {}
_active = {backend: 0 for backend in BACKENDS}
_last_used = {backend: 0.0 for backend in BACKENDS}
_lock = threading.Lock()


def now() -> float:
    return time.monotonic()


def resolve_backend(model: str | None) -> str:
    """Map a request's OpenAI-style `model` field to a backend.

    Anything naming Parakeet selects it and anything naming Whisper (including
    `whisper-1` and bare size names such as `large-v3`) selects Whisper. Other
    values, or none, fall back to ASR_BACKEND so existing callers that send
    arbitrary model names keep working.
    """
    name = (model or "").strip().lower()
    if "parakeet" in name or name == settings.PARAKEET_MODEL.lower():
        return "parakeet"
    if "whisper" in name or name == settings.MODEL_SIZE.lower() or name.startswith(
        ("tiny", "base", "small", "medium", "large", "distil")
    ):
        return "whisper"
    return settings.ASR_BACKEND


def _backend(backend: str | None) -> str:
    backend = settings.ASR_BACKEND if backend is None else backend
    if backend not in BACKENDS:
        raise ValueError(
            f"Unsupported ASR backend {backend!r}; use 'whisper' or 'parakeet'"
        )
    return backend


def model_is_loaded(backend: str | None = None) -> bool:
    with _lock:
        return _backend(backend) in _models


def mark_used(at: float | None = None, backend: str | None = None) -> None:
    with _lock:
        _last_used[_backend(backend)] = now() if at is None else at


def _load_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.MODEL_SIZE,
        device=settings.COMPUTE_DEVICE,
        compute_type="float16" if settings.COMPUTE_DEVICE == "cuda" else "int8",
    )


def _load_parakeet_model():
    import torch
    from nemo.collections.asr.models import ASRModel

    device = torch.device("cuda:0" if settings.COMPUTE_DEVICE == "cuda" else "cpu")
    model = ASRModel.from_pretrained(
        model_name=settings.PARAKEET_MODEL, map_location=device
    )
    if settings.PARAKEET_LOCAL_ATTENTION > 0:
        # Full attention grows with the square of the input: a 20-minute chunk
        # asked for 13.4 GiB more on top of 17.6 GiB and ran out of memory on
        # the 24 GB 4090. Limited-context attention, as NVIDIA recommends for
        # long audio, keeps it linear.
        context = settings.PARAKEET_LOCAL_ATTENTION
        model.change_attention_model("rel_pos_local_attn", [context, context])
        model.change_subsampling_conv_chunking_factor(1)
    return model.eval()


def _load_model(backend: str):
    if backend == "parakeet":
        return _load_parakeet_model()
    return _load_whisper_model()


def reset_model_state() -> None:
    """Drop every model and counter. Also used as a test hook."""
    with _lock:
        _models.clear()
        for backend in BACKENDS:
            _active[backend] = 0
            _last_used[backend] = 0.0
    _release_runtime_memory()


def _release_runtime_memory() -> None:
    gc.collect()
    # Parakeet runs on PyTorch, whose caching allocator keeps freed blocks
    # until asked to return them. CTranslate2 (Whisper) frees on its own.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@contextlib.contextmanager
def model_session(backend: str | None = None):
    backend = _backend(backend)
    with _lock:
        if backend not in _models:
            start = datetime.now()
            settings.logger.info(
                "Initializing ASR model backend=%s model=%s device=%s",
                backend,
                configured_model_name(backend),
                settings.COMPUTE_DEVICE,
            )
            _models[backend] = _load_model(backend)
            settings.logger.info(
                "ASR model initialized in %.2f seconds",
                (datetime.now() - start).total_seconds(),
            )
        _active[backend] += 1
        model = _models[backend]
    try:
        yield model
    finally:
        with _lock:
            _active[backend] -= 1
            _last_used[backend] = now()


def maybe_evict_idle_model(idle_timeout: float, now: float | None = None) -> bool:
    """Release every loaded model idle for idle_timeout; True if any was."""
    current = time.monotonic() if now is None else now
    evicted = []
    with _lock:
        for backend in list(_models):
            if _active[backend] > 0:
                continue
            idle_for = current - _last_used[backend]
            if idle_for < idle_timeout:
                continue
            del _models[backend]
            evicted.append((backend, idle_for))
    if not evicted:
        return False
    _release_runtime_memory()
    for backend, idle_for in evicted:
        settings.logger.info(
            "Released idle %s model after %.0fs idle", backend, idle_for
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


def configured_model_name(backend: str | None = None) -> str:
    if _backend(backend) == "parakeet":
        return settings.PARAKEET_MODEL
    return settings.MODEL_SIZE


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


def _chunk_seconds(backend: str) -> float:
    """Chunk length for the active backend; 0 means no splitting.

    Parakeet attends over the whole input at once and cannot take more than
    PARAKEET_CHUNK_SECONDS in one pass, so it always splits at that length or
    shorter, even with ASR_CHUNK_SECONDS=0.
    """
    if backend == "parakeet":
        limit = settings.PARAKEET_CHUNK_SECONDS
        return min(settings.CHUNK_SECONDS, limit) if settings.CHUNK_SECONDS > 0 else limit
    return settings.CHUNK_SECONDS


def _needs_chunking(duration: float, backend: str) -> bool:
    chunk_seconds = _chunk_seconds(backend)
    return chunk_seconds > 0 and duration > chunk_seconds


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


def _transcribe_chunked(
    transcribe_one, file_path: str, duration: float, backend: str
) -> TranscriptionResult:
    """Transcribe chunk by chunk and merge into one result.

    Language and its probability come from the first chunk; the reported
    duration is that of the original file.
    """
    chunk_dir = tempfile.mkdtemp(prefix="chunks_", dir=settings.UPLOAD_DIRECTORY)
    try:
        chunk_seconds = _chunk_seconds(backend)
        chunks = _split_audio(file_path, chunk_seconds, chunk_dir)
        settings.logger.info(
            "Transcribing %.0fs of audio in %d chunks of up to %.0fs",
            duration,
            len(chunks),
            chunk_seconds,
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


def _transcribe_whisper(model, file_path, include_word_timestamps, language, prompt=None):
    options = dict(
        beam_size=5,
        word_timestamps=include_word_timestamps,
        language=language,
    )
    if prompt:
        # Whisper copies the prompt's style. Without one, large-v3 can settle
        # into lowercase text with no punctuation for a whole stream.
        options["initial_prompt"] = prompt
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


def _transcribe_parakeet(model, file_path, include_word_timestamps, language):
    # NeMo reads audio itself and expects 16 kHz mono; uploads can be any
    # container ffmpeg understands, so convert first.
    work_dir = tempfile.mkdtemp(prefix="parakeet_", dir=settings.UPLOAD_DIRECTORY)
    try:
        wav_path = os.path.join(work_dir, "audio.wav")
        _ffmpeg_to_mono_16k(file_path, [wav_path])
        duration = _audio_duration(wav_path)
        hypothesis = model.transcribe(
            [wav_path], batch_size=1, timestamps=True, verbose=False
        )[0]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    stamps = hypothesis.timestamp or {}
    words = [
        {
            "start": item["start"],
            "end": item["end"],
            "word": item["word"],
            "probability": None,
        }
        for item in stamps.get("word", [])
    ]
    segments = []
    next_word = 0
    for item in stamps.get("segment", []):
        # NeMo builds each segment by joining consecutive words with spaces.
        count = len(item["segment"].split())
        segment_words = words[next_word:next_word + count]
        next_word += count
        segments.append(
            TranscriptionSegment(
                start=item["start"],
                end=item["end"],
                text=item["segment"],
                words=segment_words if include_word_timestamps else None,
            )
        )
    if not segments and hypothesis.text:
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=duration,
                text=hypothesis.text,
                words=words if include_word_timestamps else None,
            )
        ]
    # Parakeet does not report the language it heard, so echo the caller's.
    return TranscriptionResult(
        segments=segments,
        language=language,
        language_probability=None,
        duration=duration,
    )


async def detect_language(file_path: str, backend: str | None = None):
    backend = _backend(backend)
    if backend == "parakeet":
        raise NotImplementedError("The parakeet backend does not detect language")

    def run_whisper_detection():
        # transcribe() decodes the whole file before detecting, so for
        # long audio detect on the first chunk only.
        if not _needs_chunking(_audio_duration(file_path), backend):
            with model_session(backend) as model:
                _, info = model.transcribe(file_path)
                return info.language, info.language_probability
        chunk_dir = tempfile.mkdtemp(prefix="detect_", dir=settings.UPLOAD_DIRECTORY)
        try:
            head = os.path.join(chunk_dir, "head.wav")
            _ffmpeg_to_mono_16k(file_path, ["-t", str(settings.CHUNK_SECONDS), head])
            with model_session(backend) as model:
                _, info = model.transcribe(head)
                return info.language, info.language_probability
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    return await asyncio.to_thread(run_whisper_detection)


async def transcribe_audio(
    file_path: str,
    include_word_timestamps: bool = False,
    language: str | None = None,
    prompt: str | None = None,
    backend: str | None = None,
) -> TranscriptionResult:
    backend = _backend(backend)

    def transcribe_one(path):
        with model_session(backend) as model:
            if backend == "parakeet":
                # Parakeet has no decoder prompt; it punctuates on its own.
                return _transcribe_parakeet(
                    model, path, include_word_timestamps, language
                )
            return _transcribe_whisper(
                model, path, include_word_timestamps, language, prompt
            )

    def run_transcription():
        duration = _audio_duration(file_path)
        if _needs_chunking(duration, backend):
            return _transcribe_chunked(transcribe_one, file_path, duration, backend)
        return transcribe_one(file_path)

    return await asyncio.to_thread(run_transcription)
