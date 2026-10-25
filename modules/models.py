from pydantic import BaseModel
from typing import Optional, List

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[dict]] = None


class TranscriptionResult(BaseModel):
    segments: List[TranscriptionSegment]
    # None when the backend does not report a language (Parakeet).
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: float
