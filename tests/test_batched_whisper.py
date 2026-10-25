"""WHISPER_BATCH_SIZE switches Whisper to faster-whisper's batched pipeline.

On mlbox's RTX 4090 the sequential decoder took 331 s for an hour of film
audio with the GPU about half idle; the batched pipeline took 24 s. These tests
replace faster_whisper with fakes, so they need neither a GPU nor the model.
"""

import os
import sys
import types

os.environ.setdefault("AUTH_TOKEN", "test-secret")
sys.modules.setdefault("faster_whisper", types.ModuleType("faster_whisper"))
if not hasattr(sys.modules["faster_whisper"], "WhisperModel"):
    sys.modules["faster_whisper"].WhisperModel = object

import pytest

from modules import transcription


class Segment:
    start, end, text, words = 0.0, 1.0, " hello", []


INFO = types.SimpleNamespace(language="en", language_probability=0.99, duration=1.0)


class FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter([Segment()]), INFO


class FakeBatchedPipeline:
    instances = []

    def __init__(self, model):
        self.model = model
        self.calls = []
        FakeBatchedPipeline.instances.append(self)

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter([Segment()]), INFO


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    FakeBatchedPipeline.instances = []
    monkeypatch.setattr(
        sys.modules["faster_whisper"],
        "BatchedInferencePipeline",
        FakeBatchedPipeline,
        raising=False,
    )


def test_batch_size_zero_uses_the_sequential_decoder(monkeypatch):
    monkeypatch.setattr(transcription.settings, "WHISPER_BATCH_SIZE", 0)
    model = FakeWhisperModel()

    result = transcription._transcribe_whisper(model, "a.wav", False, None)

    assert FakeBatchedPipeline.instances == []
    assert model.calls == [
        ("a.wav", {"beam_size": 5, "word_timestamps": False, "language": None})
    ]
    assert result.segments[0].text == " hello"


def test_positive_batch_size_uses_the_batched_pipeline(monkeypatch):
    monkeypatch.setattr(transcription.settings, "WHISPER_BATCH_SIZE", 8)
    model = FakeWhisperModel()

    result = transcription._transcribe_whisper(model, "a.wav", True, "en")

    assert model.calls == []
    [pipeline] = FakeBatchedPipeline.instances
    assert pipeline.model is model
    assert pipeline.calls == [
        (
            "a.wav",
            {
                "batch_size": 8,
                "beam_size": 5,
                "word_timestamps": True,
                "language": "en",
            },
        )
    ]
    assert result.language == "en"
    assert result.segments[0].words == []


def test_prompt_seeds_the_decoder_with_punctuated_text(monkeypatch):
    # Without a prompt large-v3 kept a lowercase, unpunctuated style for a
    # whole 15-minute stream; a short punctuated prompt restores sentences.
    monkeypatch.setattr(transcription.settings, "WHISPER_BATCH_SIZE", 0)
    model = FakeWhisperModel()

    transcription._transcribe_whisper(model, "a.wav", True, "en", "Hello, everyone.")

    assert model.calls == [
        ("a.wav", {"beam_size": 5, "word_timestamps": True, "language": "en",
                   "initial_prompt": "Hello, everyone."})
    ]
