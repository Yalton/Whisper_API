import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class Settings:
    ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").strip().lower()
    MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "cpu")
    QWEN_MODEL = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
    QWEN_ALIGNER_MODEL = os.getenv(
        "QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"
    ).strip()
    QWEN_DTYPE = os.getenv("QWEN_DTYPE", "auto").strip().lower()
    QWEN_ATTN_IMPLEMENTATION = os.getenv(
        "QWEN_ATTN_IMPLEMENTATION", "sdpa"
    ).strip()
    QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "4096"))
    QWEN_MAX_BATCH_SIZE = int(os.getenv("QWEN_MAX_BATCH_SIZE", "8"))
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
    UPLOAD_DIRECTORY = "/tmp/uploaded_audio_files"
    # Seconds the model may sit unused before its VRAM is released. The next
    # request reloads it, costing ~32s for large-v3. Set to 0 to keep the model
    # resident once loaded, which is the old behaviour.
    IDLE_TIMEOUT_SECONDS = float(
        os.getenv("ASR_IDLE_TIMEOUT", os.getenv("WHISPER_IDLE_TIMEOUT", "300"))
    )
    # Audio longer than this many seconds is split into 16 kHz mono chunks of
    # this length and transcribed one chunk at a time. faster-whisper decodes
    # the whole input into RAM before it starts, so a 4-hour file peaked at
    # ~18-21 GB and was OOM-killed. Set to 0 to disable splitting.
    CHUNK_SECONDS = float(os.getenv("ASR_CHUNK_SECONDS", "1800"))
    logger = logging.getLogger()



settings = Settings()
