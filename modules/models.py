from pydantic import BaseModel
from typing import Optional, List

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[dict]] = None


class TranscriptionResult(BaseModel):
    segments: List[TranscriptionSegment]
    language: str
    language_probability: Optional[float] = None
    duration: float
