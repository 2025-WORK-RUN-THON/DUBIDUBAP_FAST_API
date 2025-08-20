## Trendy Lyrics FastAPI Backend

### Quickstart

1. Create virtualenv and install deps

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
```

2. Run the server

```bash
uvicorn app.main:app --reload
```

3. Test endpoints

- Health: `GET http://127.0.0.1:8000/health`
- Ping: `GET http://127.0.0.1:8000/api/v1/ping`

### Configuration

Environment variables are read from a local `.env` file if present. See keys in `app/core/config.py`.

### Transcription (YouTube URL → text)

- Requirements: ffmpeg installed on system and `ENABLE_TRANSCRIBE=true` in `.env`.
- macOS: `brew install ffmpeg`
- Dry-run (no download/transcribe, path planning only):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process/transcribe-url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","dry_run":true}'
```

- Real transcription (will download audio and run faster-whisper on first call):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process/transcribe-url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","model_size":"small"}'
```
