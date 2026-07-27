from .config import settings
from faster_whisper import WhisperModel
import aiofiles
import asyncio
import contextlib
import gc
import os
import threading
import time
from .models import TranscriptionSegment
from datetime import datetime


# The model is loaded on first use and released again once it has been idle for
# WHISPER_IDLE_TIMEOUT seconds. Lazy loading alone was never the problem: it was
# already lazy, but nothing ever let go, so one request pinned the weights for
# the life of the process -- 3.5 GB for large-v3, on a box used for other work.
#
# ctranslate2 3.24 exposes no unload_model on Whisper, so eviction means dropping
# every reference and letting the destructor free device memory. Measured on the
# box, the `small` model takes 1027 MiB and returns 626 MiB, the remaining
# 401 MiB being the process CUDA context, which only a process exit reclaims.
# Repeated load/unload cycles plateau exactly, so this does not leak.
_model = None
_active = 0
_last_used = 0.0
_lock = threading.Lock()


def now() -> float:
    """Monotonic clock. Wall time would let a clock change trigger eviction."""
    return time.monotonic()


def model_is_loaded() -> bool:
    with _lock:
        return _model is not None


def mark_used(at: float | None = None) -> None:
    global _last_used
    with _lock:
        _last_used = now() if at is None else at


def reset_model_state() -> None:
    """Drop the model and every counter. Test hook."""
    global _model, _active, _last_used
    with _lock:
        _model = None
        _active = 0
        _last_used = 0.0
    gc.collect()


@contextlib.contextmanager
def model_session():
    """Borrow the model, loading it if necessary, and hold off eviction.

    The load happens under the lock so that two requests arriving together
    cannot each build a model and double the VRAM they were meant to save. The
    second waits out the ~32s load and then shares the result.
    """
    global _model, _active, _last_used
    with _lock:
        if _model is None:
            start = datetime.now()
            settings.logger.info(
                "Initializing Whisper model, model_size=%s device=%s",
                settings.MODEL_SIZE,
                settings.COMPUTE_DEVICE,
            )
            _model = WhisperModel(
                settings.MODEL_SIZE,
                device=settings.COMPUTE_DEVICE,
                compute_type="float16" if settings.COMPUTE_DEVICE == "cuda" else "int8",
            )
            settings.logger.info(
                "Model initialized in %.2f seconds",
                (datetime.now() - start).total_seconds(),
            )
        _active += 1
        model = _model
    try:
        yield model
    finally:
        with _lock:
            _active -= 1
            # Idle is measured from release, not acquisition: a transcription
            # that ran for an hour has only just stopped being useful.
            _last_used = now()


def maybe_evict_idle_model(idle_timeout: float, now: float | None = None) -> bool:
    """Release the model if nothing has used it for idle_timeout seconds.

    Returns whether an eviction happened. A session in flight always wins:
    freeing the model under a running transcription would crash the request.
    """
    global _model
    with _lock:
        if _model is None or _active > 0:
            return False
        current = time.monotonic() if now is None else now
        idle_for = current - _last_used
        if idle_for < idle_timeout:
            return False
        _model = None
    # Outside the lock: the collection is the slow part, and a request arriving
    # now should be free to start loading a fresh model rather than wait on it.
    gc.collect()
    settings.logger.info(
        "Released idle Whisper model after %.0fs idle; VRAM returned to the pool",
        idle_for,
    )
    return True


async def idle_eviction_loop(
    idle_timeout: float | None = None, interval: float | None = None
) -> None:
    """Poll for an idle model and release it. Started from the app lifespan."""
    timeout = settings.IDLE_TIMEOUT_SECONDS if idle_timeout is None else idle_timeout
    if timeout <= 0:
        settings.logger.info(
            "Whisper idle eviction disabled; the model will stay resident once loaded"
        )
        return
    poll = max(5.0, min(timeout / 4.0, 60.0)) if interval is None else interval
    settings.logger.info(
        "Whisper idle eviction active: releasing the model after %.0fs idle "
        "(checked every %.0fs)",
        timeout,
        poll,
    )
    while True:
        await asyncio.sleep(poll)
        try:
            # gc.collect() can stall briefly; keep it off the event loop so it
            # cannot delay an incoming request.
            await asyncio.to_thread(maybe_evict_idle_model, timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            settings.logger.exception("Idle eviction check failed")


os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)


async def save_uploaded_file(upload_file, path: str):
    async with aiofiles.open(path, 'wb') as out_file:
        while chunk := await upload_file.read(1024 * 1024):
            await out_file.write(chunk)
    settings.logger.info(f"Saved file to {path}")


async def detect_language(file_path: str):
    settings.logger.info("Detecting language...")

    def run_detection():
        with model_session() as model:
            # faster-whisper performs language detection before returning the
            # lazy segment iterator. There is no `just_detection` parameter.
            _, info = model.transcribe(file_path)
            return info

    info = await asyncio.to_thread(run_detection)
    return info.language, info.language_probability


async def transcribe_audio(
    file_path: str,
    include_word_timestamps: bool = False,
    language: str | None = None,
):
    settings.logger.info(
        f"Transcribing audio include_word_timestamps: {include_word_timestamps} "
        f"language: {language or 'auto'}"
    )

    def run_transcription():
        # The whole session stays open until the segment generator is drained:
        # faster_whisper yields lazily, so releasing the model earlier would
        # pull it out from under the iteration.
        with model_session() as model:
            segments, info = model.transcribe(
                file_path,
                beam_size=5,
                word_timestamps=include_word_timestamps,
                language=language,
            )
            transcription_segments = [
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
            return transcription_segments, info

    transcription_segments, info = await asyncio.to_thread(run_transcription)
    settings.logger.info(f"Transcription completed. Language: {info.language}, Probability: {info.language_probability}, Duration: {info.duration}")
    return transcription_segments, info.language, info.language_probability, info.duration
