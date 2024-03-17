from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks, Depends
from typing import Optional, List
import os
from datetime import datetime
from .transcription import detect_language, save_uploaded_file, transcribe_audio
from .config import settings
from .utils import download_youtube_audio, get_auth_token


router = APIRouter()

@router.post("/v1/test/", summary="Test Endpoint", description="A simple test endpoint to check the API's responsiveness.")
async def test_api():
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        settings.logger.info(f"Current time: {current_time}")
        return {
            "message": "You hit the Whisper API, nice"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/audio/language_detection/", summary="Language Detection endpoint", description="Endpoint to detect spoken langauge in provided clip or youtube video")
async def language_detection(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None), 
    authorization: str = Depends(get_auth_token)  # Add this line
):
    if file:
        settings.logger.info(f"Received file for language detection: {file.filename}")
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, file.filename)
        await save_uploaded_file(file, file_path)
    elif youtube_url:
        settings.logger.info(f"Downloading YouTube video for language detection: {youtube_url}")
        file_path = download_youtube_audio(youtube_url, settings.UPLOAD_DIRECTORY)
    else:
        raise HTTPException(status_code=400, detail="Must provide either a file or a YouTube URL")

    language, probability = await detect_language(file_path)
    
    return {
        "language": language,
        "probability": probability
    }


@router.post("/v1/audio/transcriptions/", summary="Transcription endpoint", description="Endpoint to transcribe audio in provided clip or youtube video")
async def transcribe_upload_file(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    timestamp_granularities: Optional[str] = Form(default=""),
    response_format: Optional[str] = Form(default="raw_text"),
    authorization: str = Depends(get_auth_token)  # Add this line
):
    start_time = datetime.now()

    if file:
        settings.logger.info(f"Received file for transcription: {file.filename}")
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, file.filename)
        await save_uploaded_file(file, file_path)
    elif youtube_url:
        file_path = download_youtube_audio(youtube_url, settings.UPLOAD_DIRECTORY)
    else:
        raise HTTPException(status_code=400, detail="Must provide either a file or a YouTube URL")

    file_name = file.filename if file else youtube_url

    try:
        include_word_timestamps = "word" in timestamp_granularities
        
        transcription_segments, language, language_probability, duration = await transcribe_audio(file_path, include_word_timestamps=include_word_timestamps)
        
        transcription_text = " ".join([segment.text for segment in transcription_segments])
        

        response_data = {
            "text": transcription_text,
            "task": "transcribe",
            "language": language,
            "duration": duration,
            "language_probability": language_probability,
            "filename": file_name,
        }

        # Handling for verbose_json response format with word timestamps
        if response_format == "verbose_json":
            if include_word_timestamps:
                # Flatten words from all segments into a single list
                all_words = []
                for segment in transcription_segments:
                    if segment.words:  # Ensure words are present
                        all_words.extend(segment.words)
                response_data["words"] = all_words
            elif "segment" in timestamp_granularities:
                # Include segment information without modifying it
                response_data["segments"] = [segment.dict() for segment in transcription_segments]

        settings.logger.info(f"Transcription performed on audio sample of {duration} seconds took {(datetime.now() - start_time).total_seconds()} seconds.")
        return response_data

    except HTTPException as e:
        settings.logger.error(f"HTTP Exception: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        settings.logger.error(f"Unhandled exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")