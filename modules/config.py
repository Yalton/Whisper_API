import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class Settings:
    MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "cpu")
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
    UPLOAD_DIRECTORY = "/tmp/uploaded_audio_files"
    # Seconds the model may sit unused before its VRAM is released. The next
    # request reloads it, costing ~32s for large-v3. Set to 0 to keep the model
    # resident once loaded, which is the old behaviour.
    IDLE_TIMEOUT_SECONDS = float(os.getenv("WHISPER_IDLE_TIMEOUT", "300"))
    logger = logging.getLogger()



settings = Settings()
