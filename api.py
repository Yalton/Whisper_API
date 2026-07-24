from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from typing import Optional, List
import aiofiles
import os
from faster_whisper import WhisperModel
import logging
from datetime import datetime
import uvicorn



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Read model size and device from environment variables
model_size = os.getenv("WHISPER_MODEL_SIZE", "base")  # Default to "base" if not specified
device = os.getenv("COMPUTE_DEVICE", "cpu")  # Default to "cpu" if not specified

if device == "cuda":
    compute_type = "float16"  # Use "float16" for GPU
else:
    compute_type = "int8"  # Default to "int8" for CPU usage

start_time = datetime.now()
logger.info(f"Initializing Whisper model for transcription, model_size: {model_size}, device: {device}, compute_type: {compute_type}")
model = WhisperModel(model_size, device=device, compute_type=compute_type)
initialization_time = datetime.now() - start_time
logger.info(f"Model initialized in {initialization_time.total_seconds()} seconds; ready for requests")

# Path to save uploaded audio files temporarily
UPLOAD_DIRECTORY = "/tmp/uploaded_audio_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

async def save_uploaded_file(upload_file: UploadFile, path: str):
    async with aiofiles.open(path, 'wb') as out_file:
        content = await upload_file.read()  # Async read
        await out_file.write(content)  # Async write
    logger.info(f"Saved file to {path}")

async def transcribe_audio(file_path: str, include_word_timestamps: bool = False):
    try:
        logger.info(f"Transcribing audio include_word_timestamps: {include_word_timestamps}")
        segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=include_word_timestamps)
        
        transcription_segments = []
        if include_word_timestamps:
            for segment in segments:
                words = [{
                    "start": word.start,
                    "end": word.end,
                    "word": word.word
                } for word in segment.words]
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": words
                })
        else:
            for segment in segments:
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
        
        logger.info(f"Transcription completed. Language: {info.language}, Probability: {info.language_probability}, Duration: {info.duration}")
        
        return transcription_segments, info.language, info.language_probability, info.duration
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise e

@app.post("/test/")
async def test_api():
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Current time: {current_time}")
        return {
            "message": "You hit the Whisper API, nice"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/")
async def transcribe_upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    timestamp_granularities: Optional[str] = Form(default=""),
    response_format: Optional[str] = Form(default="raw_text")
):
    start_time = datetime.now()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Timestamp: {current_time} Received file for transcription: {file.filename}")
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    await save_uploaded_file(file, file_path)
    
    try:
        include_word_timestamps = True if timestamp_granularities == "word" else False
        
        transcription_segments, language, language_probability, duration = await transcribe_audio(file_path, include_word_timestamps=include_word_timestamps)
        
        transcription_text = " ".join([segment['text'] for segment in transcription_segments])
        
        response_data = {
            "text": transcription_text,
        }

        logger.info(f"Returning response with response_format: {response_format}, timestamp_granularities: {timestamp_granularities}")
        if response_format == "verbose_json":
            response_data.update({
                "task": "transcribe",
                "language": language,
                "duration": duration,
                "language_probability": language_probability,
                "filename": file.filename,
            })

            if timestamp_granularities == "segment":
                response_data["segments"] = transcription_segments
            elif timestamp_granularities == "word":
                response_data["words"] = transcription_segments
            else: 
                pass
        
        transcription_time = datetime.now() - start_time
        logger.info(f"Transcription perfomed on audio sample of {duration} seconds took {transcription_time.total_seconds()} seconds.")
        return response_data

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")