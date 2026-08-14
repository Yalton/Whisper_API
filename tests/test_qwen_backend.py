import asyncio
from dataclasses import dataclass

import pytest

from modules import transcription


@dataclass
class FakeTimestamp:
    text: str
    start_time: float
    end_time: float


@dataclass
class FakeResult:
    language: str
    text: str
    time_stamps: list | None


class FakeQwenModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        stamps = (
            [
                FakeTimestamp("hello", 0.1, 0.5),
                FakeTimestamp("world", 0.6, 1.0),
            ]
            if kwargs["return_time_stamps"]
            else None
        )
        return [FakeResult("English", "hello world", stamps)]


@pytest.fixture
def qwen(monkeypatch):
    previous_backend = transcription.settings.ASR_BACKEND
    previous_aligner = transcription.settings.QWEN_ALIGNER_MODEL
    transcription.settings.ASR_BACKEND = "qwen3"
    transcription.settings.QWEN_ALIGNER_MODEL = "fake-aligner"
    model = FakeQwenModel()
    monkeypatch.setattr(transcription, "_load_model", lambda: model)
    monkeypatch.setattr(transcription, "_audio_duration", lambda _: 1.2)
    transcription.reset_model_state()
    yield model
    transcription.reset_model_state()
    transcription.settings.ASR_BACKEND = previous_backend
    transcription.settings.QWEN_ALIGNER_MODEL = previous_aligner


def test_qwen_transcription_normalizes_language_and_missing_confidence(qwen):
    result = asyncio.run(transcription.transcribe_audio("speech.wav"))

    assert result.language == "en"
    assert result.language_probability is None
    assert result.duration == 1.2
    assert result.segments[0].text == "hello world"
    assert result.segments[0].words is None
    assert qwen.calls[0]["language"] is None


def test_qwen_word_timestamps_preserve_contract_with_null_probability(qwen):
    result = asyncio.run(
        transcription.transcribe_audio(
            "speech.wav", include_word_timestamps=True, language="en"
        )
    )

    assert qwen.calls[0]["language"] == "English"
    assert qwen.calls[0]["return_time_stamps"] is True
    assert result.segments[0].start == 0.1
    assert result.segments[0].end == 1.0
    assert result.segments[0].words == [
        {"start": 0.1, "end": 0.5, "word": "hello", "probability": None},
        {"start": 0.6, "end": 1.0, "word": "world", "probability": None},
    ]


def test_qwen_word_timestamps_require_aligner(qwen):
    transcription.settings.QWEN_ALIGNER_MODEL = ""

    with pytest.raises(ValueError, match="QWEN_ALIGNER_MODEL"):
        asyncio.run(
            transcription.transcribe_audio(
                "speech.wav", include_word_timestamps=True
            )
        )


def test_qwen_language_detection_uses_transcription(qwen):
    assert asyncio.run(transcription.detect_language("speech.wav")) == ("en", None)
