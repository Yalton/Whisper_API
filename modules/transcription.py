from .config import settings
from faster_whisper import WhisperModel
import aiofiles
import asyncio
import os
from .models import TranscriptionSegment
from datetime import datetime


model = None


def get_model():
    global model
    if model is None:
        start_time = datetime.now()
        settings.logger.info(
            "Initializing Whisper model, model_size=%s device=%s",
            settings.MODEL_SIZE,
            settings.COMPUTE_DEVICE,
        )
        model = WhisperModel(
            settings.MODEL_SIZE,
            device=settings.COMPUTE_DEVICE,
            compute_type="float16" if settings.COMPUTE_DEVICE == "cuda" else "int8",
        )
        settings.logger.info(
            "Model initialized in %.2f seconds",
            (datetime.now() - start_time).total_seconds(),
        )
    return model

os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)

async def save_uploaded_file(upload_file, path: str):
    async with aiofiles.open(path, 'wb') as out_file:
        while chunk := await upload_file.read(1024 * 1024):
            await out_file.write(chunk)
    settings.logger.info(f"Saved file to {path}")


async def detect_language(file_path: str):
    settings.logger.info("Detecting language...")
    _, info = await asyncio.to_thread(
        get_model().transcribe, file_path, just_detection=True
    )
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
        segments, info = get_model().transcribe(
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
