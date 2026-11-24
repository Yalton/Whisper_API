# Whisper_API

Shared conventions (`base`, `web-service`), git authority, and the verify command are declared in `.fleet.json` and auto-loaded by the global `SessionStart` hook.

## Build & Test

```bash
pytest -q
```

## Notes

- Local ASR HTTP API (`main.py`, `modules/`, `docker-compose.yml`) deployed on `mlbox`.
- See [README.md](README.md) and [QWEN3_ASR_MIGRATION.md](QWEN3_ASR_MIGRATION.md) for model/runtime details.
