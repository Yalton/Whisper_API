"""Idle eviction of the Whisper model.

The model was already loaded lazily, on the first request rather than at import.
What was missing is the other half: nothing ever released it, so a single
transcription pinned the weights for the life of the process -- 3.5 GB for
large-v3, held indefinitely on a box used for other work.

ctranslate2 3.24 exposes no unload_model on Whisper, so eviction works by
dropping every reference and letting the destructor free device memory. Measured
on the box: of 1027 MiB for the `small` model, 626 MiB returns and 401 MiB stays
as the process CUDA context. Repeated load/unload cycles plateau exactly, so
this does not leak.
"""

import os
import sys
import types

os.environ.setdefault("AUTH_TOKEN", "test-secret")

# faster_whisper must merely be importable here; the model itself is replaced at
# the transcription module's own seam below. test_contract.py installs its own
# stub under the same name, so anything relying on module identity breaks
# depending on collection order.
sys.modules.setdefault("faster_whisper", types.ModuleType("faster_whisper"))
if not hasattr(sys.modules["faster_whisper"], "WhisperModel"):
    sys.modules["faster_whisper"].WhisperModel = object

import pytest

from modules import transcription


_loads = []


class FakeWhisperModel:
    def __init__(self, size, device=None, compute_type=None):
        _loads.append(size)
        self.size = size


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    _loads.clear()
    # Patch where the name is bound, not where it came from: transcription.py
    # does `from faster_whisper import WhisperModel`, so this is the only seam
    # that holds regardless of which test module imported first.
    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    transcription.reset_model_state()
    yield
    transcription.reset_model_state()


def _later(seconds: float) -> float:
    return transcription.now() + seconds


def test_model_is_not_loaded_until_first_use():
    assert transcription.model_is_loaded() is False
    assert _loads == []


def test_first_use_loads_the_model():
    with transcription.model_session():
        pass
    assert transcription.model_is_loaded() is True
    assert len(_loads) == 1


def test_second_use_reuses_the_loaded_model():
    with transcription.model_session() as first:
        pass
    with transcription.model_session() as second:
        pass
    assert first is second
    assert len(_loads) == 1


def test_idle_model_is_evicted_once_the_timeout_passes():
    with transcription.model_session():
        pass
    assert transcription.maybe_evict_idle_model(300, now=_later(301)) is True
    assert transcription.model_is_loaded() is False


def test_model_is_kept_while_still_within_the_timeout():
    with transcription.model_session():
        pass
    assert transcription.maybe_evict_idle_model(300, now=_later(299)) is False
    assert transcription.model_is_loaded() is True


def test_model_is_never_evicted_while_a_transcription_is_in_flight():
    # A long transcription runs in a worker thread and can easily outlast the
    # idle timeout. Freeing the model under it would crash the request.
    with transcription.model_session():
        assert transcription.maybe_evict_idle_model(300, now=_later(10_000)) is False
        assert transcription.model_is_loaded() is True


def test_timeout_is_measured_from_release_not_acquisition():
    # A 40-minute transcription that just finished is not idle.
    with transcription.model_session():
        pass
    transcription.mark_used(at=_later(3600))
    assert transcription.maybe_evict_idle_model(300, now=_later(3601)) is False


def test_eviction_is_idempotent():
    with transcription.model_session():
        pass
    assert transcription.maybe_evict_idle_model(300, now=_later(301)) is True
    assert transcription.maybe_evict_idle_model(300, now=_later(302)) is False


def test_use_after_eviction_reloads_the_model():
    with transcription.model_session():
        pass
    transcription.maybe_evict_idle_model(300, now=_later(301))
    with transcription.model_session():
        pass
    assert transcription.model_is_loaded() is True
    assert len(_loads) == 2


def test_concurrent_sessions_load_the_model_only_once():
    with transcription.model_session() as a:
        with transcription.model_session() as b:
            assert a is b
    assert len(_loads) == 1


def test_nested_sessions_keep_the_model_until_the_outermost_exits():
    with transcription.model_session():
        with transcription.model_session():
            pass
        # The inner session ended, but the outer one is still working.
        assert transcription.maybe_evict_idle_model(0, now=_later(10_000)) is False
    assert transcription.maybe_evict_idle_model(0, now=_later(10_001)) is True


def test_transcribe_audio_releases_the_model_for_eviction(tmp_path):
    # The end-to-end shape: a real transcription must leave the model
    # evictable, or the loop can never reclaim anything.
    import asyncio

    class Segment:
        start, end, text, words = 0.0, 1.0, " hi", []

    class Info:
        language, language_probability, duration = "en", 0.99, 1.0

    def fake_transcribe(self, *args, **kwargs):
        return iter([Segment()]), Info()

    FakeWhisperModel.transcribe = fake_transcribe
    asyncio.run(transcription.transcribe_audio(str(tmp_path / "a.wav")))
    assert transcription.model_is_loaded() is True
    assert transcription.maybe_evict_idle_model(300, now=_later(301)) is True


def test_detect_language_uses_supported_transcribe_contract(tmp_path):
    import asyncio

    class Info:
        language, language_probability = "en", 0.99

    calls = []

    def fake_transcribe(self, file_path, **kwargs):
        calls.append((file_path, kwargs))
        return iter(()), Info()

    FakeWhisperModel.transcribe = fake_transcribe
    audio_path = str(tmp_path / "a.wav")

    assert asyncio.run(transcription.detect_language(audio_path)) == ("en", 0.99)
    assert calls == [(audio_path, {})]
    assert transcription.maybe_evict_idle_model(300, now=_later(301)) is True
