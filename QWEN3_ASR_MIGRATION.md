# Whisper to Qwen3-ASR Migration Reference

## Purpose

This service is a local automatic speech recognition (ASR), or speech-to-text,
API. It is not a text-to-speech (TTS) service.

The service now supports two interchangeable inference backends:

- `whisper`: faster-whisper, retained for compatibility and as a fallback.
- `qwen3`: Qwen3-ASR, intended for improved multilingual, accented, noisy, and
  otherwise difficult audio.

The migration preserves the existing HTTP endpoints so consuming applications
do not need to adopt a new URL, authentication method, or upload format.

## Compatibility Summary

The following API behavior is unchanged:

- `POST /v1/audio/transcriptions/`
- `POST /v1/audio/language_detection/`
- Bearer-token authentication
- Legacy raw authorization tokens
- Multipart file uploads
- YouTube URL transcription
- Optional language selection
- `raw_text` and `verbose_json` response modes
- `word` and `segment` timestamp request fields
- Temporary upload cleanup

Deploying with `ASR_BACKEND=whisper` retains the previous model behavior and
response value types. This is the recommended first deployment step.

Switching to `ASR_BACKEND=qwen3` retains the endpoints and overall response
structure, but introduces the model-specific differences described below.

## Existing Transcription Request

No client-side request changes are required:

```bash
curl http://server:8001/v1/audio/transcriptions/ \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@speech.wav"
```

Word and segment timestamps can still be requested using the OpenAI-style
multipart field:

```bash
curl http://server:8001/v1/audio/transcriptions/ \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@speech.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment"
```

The legacy comma-separated field remains supported:

```text
timestamp_granularities=word,segment
```

## Response Contract

A normal response retains these fields:

```json
{
  "text": "Recognized speech",
  "task": "transcribe",
  "language": "en",
  "duration": 2.4,
  "language_probability": 0.99,
  "filename": "speech.wav",
  "model": "large-v3"
}
```

Verbose responses continue to include `segments` and `words` when requested.

### Qwen3-ASR Differences

Qwen3-ASR does not expose confidence measurements equivalent to Whisper's
language probability and per-word probability. The service does not invent
confidence values. Under the Qwen backend:

```json
{
  "language_probability": null,
  "words": [
    {
      "start": 0.48,
      "end": 0.88,
      "word": "Hello",
      "probability": null
    }
  ]
}
```

Clients must therefore treat these fields as nullable:

- `language_probability`: `number | null`
- `words[].probability`: `number | null`

Qwen currently returns one top-level segment containing the complete
transcript. When word timestamps are requested, aligned word-like units appear
inside that segment and in the top-level `words` collection. Clients must not
assume that every response contains multiple Whisper-style segments.

The Qwen forced aligner produces words for languages using space-delimited text
and character-like units for some other writing systems. Clients should treat
the `word` property as an aligned text unit rather than requiring it to be a
Western-language word.

## Client Compatibility Checklist

Before changing the server to Qwen, verify each consuming project:

- Does not require `language_probability` to be a non-null number.
- Does not require `words[].probability` to be a non-null number.
- Does not assume that `segments` contains multiple elements.
- Uses the `text` field as the authoritative complete transcript.
- Treats the `model` field as descriptive instead of requiring `large-v3`.
- Accepts Qwen model names such as `Qwen/Qwen3-ASR-1.7B`.
- Does not reject additional languages or mixed-language identifiers.
- Uses request timeouts long enough for a cold model load.

An application that only consumes `text`, `language`, `duration`, and
`filename` normally requires no changes.

### Example TypeScript Types

```typescript
type TranscriptionWord = {
  start: number;
  end: number;
  word: string;
  probability: number | null;
};

type TranscriptionSegment = {
  start: number;
  end: number;
  text: string;
  words: TranscriptionWord[] | null;
};

type TranscriptionResponse = {
  text: string;
  task: "transcribe";
  language: string;
  duration: number;
  language_probability: number | null;
  filename: string;
  model: string;
  segments?: TranscriptionSegment[];
  words?: TranscriptionWord[];
};
```

### Example Python Handling

```python
result = response.json()

text = result["text"]
language = result["language"]

language_confidence = result.get("language_probability")
if language_confidence is not None:
    print(f"Detected {language} with confidence {language_confidence:.2%}")
else:
    print(f"Detected {language}; this backend does not report confidence")

for word in result.get("words", []):
    confidence = word.get("probability")
    # Do not compare confidence with a numeric threshold unless it is present.
```

## Server Configuration

### Compatibility Deployment

Use Whisper during the initial deployment:

```bash
export AUTH_TOKEN="replace-with-the-existing-token"
export ASR_BACKEND="whisper"
export WHISPER_MODEL_SIZE="large-v3"
export COMPUTE_DEVICE="cuda"
export ASR_IDLE_TIMEOUT="300"

docker compose build
docker compose up -d
```

This upgrades the application and faster-whisper dependency without changing
the selected backend.

### Qwen3-ASR Deployment

Recommended configuration for the RTX 4090 host:

```bash
export AUTH_TOKEN="replace-with-the-existing-token"
export ASR_BACKEND="qwen3"
export QWEN_ASR_MODEL="Qwen/Qwen3-ASR-1.7B"
export QWEN_ALIGNER_MODEL="Qwen/Qwen3-ForcedAligner-0.6B"
export QWEN_DTYPE="auto"
export QWEN_ATTN_IMPLEMENTATION="sdpa"
export QWEN_MAX_NEW_TOKENS="4096"
export QWEN_MAX_BATCH_SIZE="8"
export COMPUTE_DEVICE="cuda"
export ASR_IDLE_TIMEOUT="300"

docker compose up -d
```

`QWEN_DTYPE=auto` selects BF16 on the RTX 4090. `sdpa` uses PyTorch's built-in
scaled-dot-product attention and does not require compiling FlashAttention.

Model weights are downloaded on the first request unless they already exist in
the container's Hugging Face cache. The first request will consequently take
longer.

### Running Qwen Without Timestamps

If no consuming application requires word timestamps, the forced aligner can
be disabled:

```bash
export QWEN_ALIGNER_MODEL=""
```

This reduces model loading time and memory use. Requests for word timestamps
will fail until the aligner is configured again.

## Deployment Procedure

1. Record the current Git revision, environment variables, container status,
   and image ID.
2. Preserve the existing `AUTH_TOKEN` and published port.
3. Deploy the new image with `ASR_BACKEND=whisper`.
4. Run the compatibility tests below.
5. Test every known consuming project.
6. Change only `ASR_BACKEND` to `qwen3` and configure the Qwen model and
   aligner.
7. Allow the first request to download and initialize the models.
8. Repeat the same compatibility tests.
9. Compare representative Whisper and Qwen transcripts before making Qwen the
   normal production backend.

Do not delete the Whisper configuration or image during the evaluation period.

## Post-Deployment Validation

### Health Test

```bash
curl -X POST http://server:8001/v1/test/
```

### Basic Transcription

```bash
curl http://server:8001/v1/audio/transcriptions/ \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@speech.wav"
```

Confirm:

- The response is HTTP 200.
- `text` contains the expected transcript.
- `language` contains an expected language identifier.
- `duration` is close to the audio duration.
- `filename` matches the uploaded filename.
- `model` identifies the configured model.

### Timestamp Transcription

```bash
curl http://server:8001/v1/audio/transcriptions/ \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@speech.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment"
```

Confirm:

- `segments` is present.
- `words` is present and non-empty.
- Word start and end times are ordered.
- Qwen probability values are `null`, not fabricated numeric values.

### Language Detection

```bash
curl http://server:8001/v1/audio/language_detection/ \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@speech.wav"
```

Under Qwen, the `probability` value is `null`.

### Authentication

Confirm that a missing or incorrect token returns HTTP 401 and that the
existing production token still succeeds.

## Model Eviction and Cold Starts

Both backends load lazily on the first request. With the default
`ASR_IDLE_TIMEOUT=300`, model weights are released after five idle minutes so
other GPU workloads can reclaim VRAM.

The next request incurs another model-load delay. Applications should use a
request timeout that accommodates this cold start. Set the timeout to `0` to
keep the selected model resident:

```bash
export ASR_IDLE_TIMEOUT="0"
```

Keeping Qwen3-ASR and its aligner resident improves latency but reserves GPU
memory continuously.

## Rollback

The fastest model-level rollback does not require reverting code:

```bash
export ASR_BACKEND="whisper"
export WHISPER_MODEL_SIZE="large-v3"
docker compose up -d
```

If a full image rollback is necessary:

1. Restore the previously recorded Git revision or image ID.
2. Restore the prior environment variables.
3. Recreate the container.
4. Run the basic transcription and authentication tests.

Do not change the API port or authentication token as part of a model rollback.
Keeping those values stable prevents an inference rollback from becoming a
client configuration migration.

## Known Limitations

- Qwen confidence values are unavailable and represented as `null`.
- Qwen top-level segmentation differs from Whisper segmentation.
- Qwen word timestamps require the separate forced-aligner model.
- Forced alignment supports fewer languages than Qwen transcription.
- Cold-start latency is greater when both the ASR model and aligner must load.
- The current implementation uses the Qwen Transformers backend, not vLLM
  streaming.

## Local Validation Completed

Before production deployment, the implementation was validated with:

- 23 passing automated API, backend, and lifecycle tests.
- A successful CUDA 12.4/PyTorch 2.6 Docker build.
- Successful imports of Qwen3-ASR and faster-whisper in the built image.
- Real Qwen3-ASR-0.6B FP16 inference on an RTX 2070.
- Real Qwen forced alignment producing ordered word timestamps.

The production deployment should still be tested with Qwen3-ASR-1.7B on the
RTX 4090 and with representative audio from the actual consuming projects.
