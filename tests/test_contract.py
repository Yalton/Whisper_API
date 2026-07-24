import os
import sys
import types

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


def make_client(tmp_path, monkeypatch):
    async def fake_transcribe_audio(_, include_word_timestamps=False):
        assert include_word_timestamps is True
        return (
            [
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
            "en",
            0.99,
            1.0,
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
    assert body["language"] == "en"
    assert body["segments"][0]["words"][0]["word"] == " hello"
    assert body["words"][1]["probability"] == 0.8
    assert list(tmp_path.iterdir()) == []


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
