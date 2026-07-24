import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class Settings:
    MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "cpu")
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
    UPLOAD_DIRECTORY = "/tmp/uploaded_audio_files"
    logger = logging.getLogger()



settings = Settings()
