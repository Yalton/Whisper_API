import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class Settings:
    # Backend for requests whose `model` field names neither Whisper nor
    # Parakeet: `whisper` or `parakeet`.
    ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").strip().lower()
    MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "cpu")
    PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    # Parakeet transcribes at most 20 minutes in one pass, so longer audio is
    # always split into chunks no longer than this.
    PARAKEET_CHUNK_SECONDS = float(os.getenv("PARAKEET_CHUNK_SECONDS", "1200"))
    # Frames of left and right context for Parakeet's limited-context attention
    # (one frame is 80 ms). 0 keeps full attention, which needs more VRAM than
    # a 24 GB card has for a 20-minute chunk.
    PARAKEET_LOCAL_ATTENTION = int(os.getenv("PARAKEET_LOCAL_ATTENTION", "256"))
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
    # Whisper only: >0 runs faster-whisper's BatchedInferencePipeline, which
    # splits audio on speech (Silero VAD) and decodes that many segments per
    # GPU batch. 0 keeps the sequential decoder.
    WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "0"))
    logger = logging.getLogger()



settings = Settings()
