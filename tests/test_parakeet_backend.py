"""NVIDIA Parakeet backend, selected per request alongside Whisper.

Parakeet takes at most 20 minutes of audio in one pass, so the backend always
splits longer audio at PARAKEET_CHUNK_SECONDS, even when ASR_CHUNK_SECONDS is
larger or disabled. These tests use real ffmpeg on generated audio and a fake
NeMo model with a 2-second chunk length.
"""

import asyncio
import os
import shutil
import subprocess
import types

os.environ.setdefault("AUTH_TOKEN", "test-secret")

import pytest

from modules import transcription

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)


class FakeParakeetModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        path = audio[0]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.calls.append((transcription._audio_duration(path), probe, kwargs))
        n = len(self.calls)
        timestamp = {
            "word": [
                {"word": "Hello", "start": 0.25, "end": 0.5},
                {"word": "there.", "start": 0.5, "end": 0.75},
                {"word": f"Chunk{n}.", "start": 1.0, "end": 1.5},
            ],
            "segment": [
                {"segment": "Hello there.", "start": 0.25, "end": 0.75},
                {"segment": f"Chunk{n}.", "start": 1.0, "end": 1.5},
            ],
        }
        return [types.SimpleNamespace(text=f"Hello there. Chunk{n}.", timestamp=timestamp)]


@pytest.fixture
def parakeet(tmp_path, monkeypatch):
    model = FakeParakeetModel()
    # Whisper stays the default; these requests ask for Parakeet explicitly.
    monkeypatch.setattr(transcription.settings, "ASR_BACKEND", "whisper")
    monkeypatch.setattr(transcription.settings, "PARAKEET_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(transcription.settings, "CHUNK_SECONDS", 1800.0)
    monkeypatch.setattr(transcription.settings, "UPLOAD_DIRECTORY", str(tmp_path))
    def load(backend):
        assert backend == "parakeet"
        return model

    monkeypatch.setattr(transcription, "_load_model", load)
    transcription.reset_model_state()
    yield model
    transcription.reset_model_state()


def make_audio(path, seconds):
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100",
            "-ac", "2", str(path),
        ],
        check=True,
    )
    return str(path)


def test_parakeet_caps_chunks_below_the_general_chunk_length(parakeet, tmp_path):
    audio = make_audio(tmp_path / "long.flac", 5)

    result = asyncio.run(
        transcription.transcribe_audio(
            audio, include_word_timestamps=True, backend="parakeet"
        )
    )

    durations = [duration for duration, _, _ in parakeet.calls]
    assert len(durations) == 3
    # The segment muxer cuts on packet boundaries, a fraction of a second past
    # the chunk length at most.
    assert max(durations) <= 2.0 + 0.15
    assert {probe for _, probe, _ in parakeet.calls} == {"16000,1"}
    assert all(kwargs["timestamps"] is True for _, _, kwargs in parakeet.calls)
    assert [s.text for s in result.segments] == [
        "Hello there.", "Chunk1.", "Hello there.", "Chunk2.", "Hello there.", "Chunk3.",
    ]
    offset = durations[0]
    assert result.segments[3].start == pytest.approx(offset + 1.0)
    assert result.segments[2].words == [
        {"start": pytest.approx(offset + 0.25), "end": pytest.approx(offset + 0.5),
         "word": "Hello", "probability": None},
        {"start": pytest.approx(offset + 0.5), "end": pytest.approx(offset + 0.75),
         "word": "there.", "probability": None},
    ]
    assert result.duration == pytest.approx(5.0, abs=0.05)
    assert sorted(os.listdir(tmp_path)) == ["long.flac"]


def test_parakeet_splits_even_when_general_chunking_is_disabled(parakeet, tmp_path, monkeypatch):
    monkeypatch.setattr(transcription.settings, "CHUNK_SECONDS", 0.0)
    audio = make_audio(tmp_path / "long.flac", 5)

    asyncio.run(transcription.transcribe_audio(audio, backend="parakeet"))

    assert len(parakeet.calls) == 3


def test_parakeet_uses_the_shorter_general_chunk_length(parakeet, tmp_path, monkeypatch):
    monkeypatch.setattr(transcription.settings, "CHUNK_SECONDS", 1.0)
    audio = make_audio(tmp_path / "long.flac", 3)

    asyncio.run(transcription.transcribe_audio(audio, backend="parakeet"))

    assert len(parakeet.calls) == 3


def test_parakeet_short_audio_is_one_pass_without_words(parakeet, tmp_path):
    audio = make_audio(tmp_path / "short.flac", 1)

    result = asyncio.run(transcription.transcribe_audio(audio, language="en", backend="parakeet"))

    assert len(parakeet.calls) == 1
    assert parakeet.calls[0][1] == "16000,1"
    assert [s.text for s in result.segments] == ["Hello there.", "Chunk1."]
    assert all(s.words is None for s in result.segments)
    assert result.language == "en"
    assert result.language_probability is None
    assert sorted(os.listdir(tmp_path)) == ["short.flac"]


def test_parakeet_reports_no_language_when_none_requested(parakeet, tmp_path):
    audio = make_audio(tmp_path / "short.flac", 1)

    result = asyncio.run(transcription.transcribe_audio(audio, backend="parakeet"))

    assert result.language is None


def test_parakeet_language_detection_is_unsupported(parakeet):
    with pytest.raises(NotImplementedError):
        asyncio.run(transcription.detect_language("speech.wav", backend="parakeet"))


def test_parakeet_model_name_is_reported(parakeet, monkeypatch):
    monkeypatch.setattr(transcription.settings, "PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

    assert transcription.configured_model_name("parakeet") == "nvidia/parakeet-tdt-0.6b-v3"


def test_whisper_and_parakeet_stay_loaded_side_by_side(monkeypatch):
    loaded = []

    def load(backend):
        loaded.append(backend)
        return object()

    monkeypatch.setattr(transcription, "_load_model", load)
    transcription.reset_model_state()
    try:
        with transcription.model_session("whisper") as whisper:
            with transcription.model_session("parakeet") as parakeet:
                assert whisper is not parakeet
        assert loaded == ["whisper", "parakeet"]
        assert transcription.model_is_loaded("whisper")
        assert transcription.model_is_loaded("parakeet")

        # Each model's idle clock is its own: only the idle one is released.
        later = transcription.now() + 1000
        transcription.mark_used(at=later, backend="whisper")
        assert transcription.maybe_evict_idle_model(300, now=later + 1) is True
        assert transcription.model_is_loaded("whisper")
        assert not transcription.model_is_loaded("parakeet")
    finally:
        transcription.reset_model_state()
