from pydantic import BaseModel
from typing import Optional, List

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[dict]] = None
