"""Long audio is transcribed in chunks so host RAM stays bounded.

faster-whisper decodes the whole input before it starts; a 4-hour upload peaked
at ~18-21 GB and the kernel OOM-killed the worker. These tests use real ffmpeg
on a few seconds of generated audio with a 2-second chunk length.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import types

os.environ.setdefault("AUTH_TOKEN", "test-secret")
sys.modules.setdefault("faster_whisper", types.ModuleType("faster_whisper"))
if not hasattr(sys.modules["faster_whisper"], "WhisperModel"):
    sys.modules["faster_whisper"].WhisperModel = object

import pytest

from modules import transcription

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)


class Word:
    def __init__(self, start, end, word):
        self.start, self.end, self.word, self.probability = start, end, word, 0.9


class Segment:
    def __init__(self, text):
        self.start, self.end, self.text = 0.25, 1.5, text
        self.words = [Word(0.25, 0.75, " a"), Word(1.0, 1.5, " b")]


class FakeWhisperModel:
    def __init__(self):
        self.paths = []
        self.durations = []

    def transcribe(self, path, **kwargs):
        self.paths.append(path)
        self.durations.append(transcription._audio_duration(path))
        info = types.SimpleNamespace(
            language="en" if len(self.paths) == 1 else "de",
            language_probability=0.9,
            duration=transcription._audio_duration(path),
        )
        return iter([Segment(f" chunk{len(self.paths)}")]), info


@pytest.fixture
def whisper(tmp_path, monkeypatch):
    model = FakeWhisperModel()
    monkeypatch.setattr(transcription.settings, "ASR_BACKEND", "whisper")
    monkeypatch.setattr(transcription.settings, "CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(transcription.settings, "UPLOAD_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(transcription, "_load_model", lambda: model)
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


def test_long_audio_is_split_and_timestamps_are_offset(whisper, tmp_path):
    audio = make_audio(tmp_path / "long.flac", 5)

    result = asyncio.run(
        transcription.transcribe_audio(audio, include_word_timestamps=True)
    )

    assert len(whisper.paths) == 3
    assert all(path.endswith(".wav") for path in whisper.paths)
    assert [s.text for s in result.segments] == [" chunk1", " chunk2", " chunk3"]
    # The segment muxer cuts near, not exactly at, the chunk length; offsets are
    # the measured durations of the preceding chunks.
    offsets = [s.start - 0.25 for s in result.segments]
    assert offsets[0] == pytest.approx(0.0)
    assert offsets == pytest.approx([0.0, 2.0, 4.0], abs=0.15)
    assert offsets == sorted(offsets)
    assert [w["start"] for w in result.segments[1].words] == pytest.approx(
        [offsets[1] + 0.25, offsets[1] + 1.0]
    )
    assert result.duration == pytest.approx(5.0, abs=0.05)
    # Chunks are contiguous: offsets are cumulative and cover the whole file.
    assert offsets[2] == pytest.approx(sum(whisper.durations[:2]))
    assert sum(whisper.durations) == pytest.approx(5.0, abs=0.05)
    # Language comes from the first chunk.
    assert result.language == "en"
    # The upload itself is the route's to delete; every chunk is gone.
    assert sorted(os.listdir(tmp_path)) == ["long.flac"]


def test_chunks_are_mono_16k(whisper, tmp_path):
    audio = make_audio(tmp_path / "long.flac", 5)
    seen = []
    original = whisper.transcribe

    def probe(path, **kwargs):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        seen.append(out)
        return original(path, **kwargs)

    whisper.transcribe = probe
    asyncio.run(transcription.transcribe_audio(audio))
    assert set(seen) == {"16000,1"}


def test_short_audio_is_transcribed_whole(whisper, tmp_path):
    audio = make_audio(tmp_path / "short.flac", 1)

    result = asyncio.run(transcription.transcribe_audio(audio))

    assert whisper.paths == [audio]
    assert result.segments[0].start == pytest.approx(0.25)


def test_chunk_files_are_removed_when_transcription_fails(whisper, tmp_path):
    audio = make_audio(tmp_path / "long.flac", 5)

    def boom(path, **kwargs):
        raise RuntimeError("model failed")

    whisper.transcribe = boom
    with pytest.raises(RuntimeError):
        asyncio.run(transcription.transcribe_audio(audio))
    assert sorted(os.listdir(tmp_path)) == ["long.flac"]


def test_chunking_can_be_disabled(whisper, tmp_path, monkeypatch):
    monkeypatch.setattr(transcription.settings, "CHUNK_SECONDS", 0.0)
    audio = make_audio(tmp_path / "long.flac", 5)

    asyncio.run(transcription.transcribe_audio(audio))

    assert whisper.paths == [audio]


def test_language_detection_on_long_audio_uses_only_the_head(whisper, tmp_path):
    audio = make_audio(tmp_path / "long.flac", 5)

    language, _ = asyncio.run(transcription.detect_language(audio))

    assert language == "en"
    assert len(whisper.paths) == 1
    assert whisper.paths[0] != audio
    assert sorted(os.listdir(tmp_path)) == ["long.flac"]


def test_startup_purge_removes_leftover_uploads_and_chunk_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription.settings, "UPLOAD_DIRECTORY", str(tmp_path))
    (tmp_path / "orphan.flac").write_bytes(b"x")
    (tmp_path / "chunks_abc").mkdir()
    (tmp_path / "chunks_abc" / "chunk_00000.wav").write_bytes(b"x")

    assert transcription.purge_upload_directory() == 2
    assert list(tmp_path.iterdir()) == []
