from .config import settings
from faster_whisper import WhisperModel
import aiofiles
import os
from .models import TranscriptionSegment
from datetime import datetime


start_time = datetime.now()
settings.logger.info(f"Initializing Whisper model for transcription, model_size: {settings.MODEL_SIZE}, device: {settings.COMPUTE_DEVICE}")

model = WhisperModel(settings.MODEL_SIZE, device=settings.COMPUTE_DEVICE, compute_type="float16" if settings.COMPUTE_DEVICE == "cuda" else "int8")

initialization_time = datetime.now() - start_time
settings.logger.info(f"Model initialized in {initialization_time.total_seconds()} seconds; ready for requests")

os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)

async def save_uploaded_file(upload_file, path: str):
    async with aiofiles.open(path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    settings.logger.info(f"Saved file to {path}")


async def detect_language(file_path: str):
    settings.logger.info("Detecting language...")
    _, info = model.transcribe(file_path, just_detection=True) 
    return info.language, info.language_probability

async def transcribe_audio(file_path: str, include_word_timestamps: bool = False):
    settings.logger.info(f"Transcribing audio include_word_timestamps: {include_word_timestamps}")
    segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=include_word_timestamps)
    transcription_segments = [TranscriptionSegment(
        start=segment.start,
        end=segment.end,
        text=segment.text,
        words=[{"start": word.start, "end": word.end, "word": word.word} for word in segment.words] if include_word_timestamps else None
    ) for segment in segments]
    settings.logger.info(f"Transcription completed. Language: {info.language}, Probability: {info.language_probability}, Duration: {info.duration}")
    return transcription_segments, info.language, info.language_probability, info.duration
