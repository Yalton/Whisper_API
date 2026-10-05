import os
import sys
import types

import pytest

os.environ["AUTH_TOKEN"] = "test-secret"

# The HTTP contract tests do not load a real model.
fake_faster_whisper = types.ModuleType("faster_whisper")
fake_faster_whisper.WhisperModel = object
sys.modules.setdefault("faster_whisper", fake_faster_whisper)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules import routes
from modules.config import settings
from modules.models import TranscriptionSegment
from modules.models import TranscriptionResult


def make_client(tmp_path, monkeypatch, seen_languages=None):
    async def fake_transcribe_audio(_, include_word_timestamps=False, language=None):
        assert include_word_timestamps is True
        if seen_languages is not None:
            seen_languages.append(language)
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start=0,
                    end=1,
                    text=" hello world",
                    words=[
                        {
                            "start": 0,
                            "end": 0.4,
                            "word": " hello",
                            "probability": 0.9,
                        },
                        {
                            "start": 0.5,
                            "end": 1,
                            "word": " world",
                            "probability": 0.8,
                        },
                    ],
                )
            ],
            language="en",
            language_probability=0.99,
            duration=1.0,
        )

    monkeypatch.setattr(routes, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(settings, "UPLOAD_DIRECTORY", str(tmp_path))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_openai_compatible_verbose_response(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post(
        "/v1/audio/transcriptions/",
        headers={"Authorization": "Bearer test-secret"},
        files={"file": ("speech.flac", b"fixture", "audio/flac")},
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "text",
        "task",
        "language",
        "duration",
        "language_probability",
        "filename",
        "model",
        "segments",
        "words",
    }
    assert body["language"] == "en"
    assert body["segments"][0]["words"][0]["word"] == " hello"
    assert body["words"][1]["probability"] == 0.8
    assert list(tmp_path.iterdir()) == []


def test_language_form_field_pins_transcription_language(tmp_path, monkeypatch):
    seen_languages = []
    client = make_client(tmp_path, monkeypatch, seen_languages)
    response = client.post(
        "/v1/audio/transcriptions/",
        headers={"Authorization": "Bearer test-secret"},
        files={"file": ("speech.flac", b"fixture", "audio/flac")},
        data={
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
            "language": "en",
        },
    )
    assert response.status_code == 200
    assert seen_languages == ["en"]


def test_language_defaults_to_auto_detection(tmp_path, monkeypatch):
    seen_languages = []
    client = make_client(tmp_path, monkeypatch, seen_languages)
    response = client.post(
        "/v1/audio/transcriptions/",
        headers={"Authorization": "Bearer test-secret"},
        files={"file": ("speech.flac", b"fixture", "audio/flac")},
        data={
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        },
    )
    assert response.status_code == 200
    assert seen_languages == [None]


def test_auth_accepts_legacy_raw_token(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post(
        "/v1/audio/transcriptions/",
        headers={"Authorization": "test-secret"},
        files={"file": ("speech.flac", b"fixture", "audio/flac")},
        data={
            "response_format": "verbose_json",
            "timestamp_granularities": "word",
        },
    )
    assert response.status_code == 200


def test_auth_rejects_bad_token(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post(
        "/v1/audio/transcriptions/",
        headers={"Authorization": "Bearer wrong"},
        files={"file": ("speech.flac", b"fixture", "audio/flac")},
    )
    assert response.status_code == 401


def test_language_detection_returns_metadata_and_removes_upload(tmp_path, monkeypatch):
    seen_paths = []

    async def fake_detect_language(path):
        seen_paths.append(path)
        assert os.path.exists(path)
        return "en", 0.98

    monkeypatch.setattr(routes, "detect_language", fake_detect_language)
    monkeypatch.setattr(settings, "UPLOAD_DIRECTORY", str(tmp_path))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/language_detection/",
        headers={"Authorization": "Bearer test-secret"},
        files={"file": ("../../speech.flac", b"fixture", "audio/flac")},
    )

    assert response.status_code == 200
    assert response.json() == {"language": "en", "probability": 0.98}
    assert len(seen_paths) == 1
    assert os.path.dirname(seen_paths[0]) == str(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_upload_save_leaves_no_partial_file(tmp_path, monkeypatch):
    async def failing_save(upload_file, path):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise OSError("disk full")

    client = make_client(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "save_uploaded_file", failing_save)
    with pytest.raises(OSError):
        client.post(
            "/v1/audio/transcriptions/",
            headers={"Authorization": "Bearer test-secret"},
            files={"file": ("speech.flac", b"fixture", "audio/flac")},
        )
    assert list(tmp_path.iterdir()) == []
