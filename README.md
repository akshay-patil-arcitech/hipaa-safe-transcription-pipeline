# Live Analysis Pipeline

Local speaker diarization plus transcription for audio and video files, with optional PII/PHI masking and role labeling. The primary CLI entry point is `main.py`. FastAPI endpoints are available via `api.py`, and a WhisperX-based workflow is documented in `WORKFLOW.md`.

## Features

- Speaker diarization (Picovoice Falcon) with post-processing
- Transcription (OpenAI Whisper)
- Optional PII/PHI masking (Presidio plus regex fallback)
- Speaker role mapping by duration
- JSON outputs with timestamps
- FastAPI for file upload and processing

## Requirements

- Python 3.10+ (tested on 3.12)
- `ffmpeg` available on `PATH`
- Picovoice AccessKey in `.env` as `PICOVOICE_ACCESS_KEY`
- Optional: CUDA-capable GPU for faster Whisper

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you will use masking, install the spaCy model(s) you need:

```powershell
python -m spacy download en_core_web_sm
```

## Quick Start (CLI)

```powershell
python main.py --input "C:\path\to\audio_or_video.ext"
```

Common options:

- `--model` Whisper model name (default: `small`)
- `--device` `auto`, `cpu`, or `cuda`
- `--language` force language code (e.g., `en`)
- `--fast` fastest decoding settings
- `--merge-gap` merge same-speaker segments within this gap (sec)
- `--min-seg` minimum segment duration (sec)
- `--no-collapse-short` keep short segments separate
- `--pad-sec` pad segment audio on both sides (sec)
- `--max-speakers` keep only the top-N speakers (post-process)
- `--min-speaker-total` keep speakers with at least this many total seconds (post-process)
- `--masking` enable Presidio masking for PII/PHI
- `--masking-entities` comma-separated Presidio entities to mask. Supports `PII`, `PHI`, and `HIPAA` shortcuts
- `--masking-blocklist` comma-separated phrases to mask. Use `TAG:phrase` to set tag (e.g., `AGENCY:Name,PERSON:John`)
- `--masking-context-rules` enable contextual HIPAA rules (diagnosis plus gestational age)
- `--speaker-roles` comma-separated roles assigned by speaker duration (e.g., `Doctor,Interpreter,Patient,Partner`)

## Output

CLI outputs in the current working directory:

- JSON file: `<input_stem>_transcribes.json`
- Log file: `diarization_transcription.log`

Each JSON entry includes:

- `speaker`: `A`, `B`, `C`, ...
- `transcribe`: transcription text
- `start_ts` / `end_ts`: timestamps

## API (FastAPI)

Run the server:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Example request:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/transcribe" -Method Post -Form @{
  file = Get-Item "C:\path\to\audio_or_video.ext"
  model = "small"
  device = "auto"
  masking = $true
  masking_entities = "PERSON,PHONE_NUMBER,EMAIL_ADDRESS,US_SSN,DATE_TIME,LOCATION"
  masking_blocklist = "Language World Services,Some Clinic Name"
  masking_context_rules = $true
  max_speakers = 4
  min_speaker_total = 6
  speaker_roles = "Doctor,Interpreter,Patient,Partner"
}
```

API note: the server writes incremental progress to `<upload_stem>_transcripts.json` in the working directory while processing. The response includes only the file name (no full path).

## PII/PHI Masking (Presidio)

When `--masking` is enabled, entities are replaced with `[REDACTED]`. Presidio uses spaCy models per language. If a model is missing for a language, the system falls back to regex-only masking.

Example model installs:

```powershell
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download ar_core_news_sm
python -m spacy download vi_core_news_sm
python -m spacy download zh_core_web_sm
```

Notes:
- Haitian Creole (`ht`) and Cantonese (`yue`) do not have official spaCy models and will use regex-only masking.
- Mandarin uses `zh_core_web_sm` (Chinese).

## Related Docs

- `DOCS.md` for system overview, benchmarks, and accuracy notes
- `WORKFLOW.md` for the WhisperX-based pipeline and API details

## Troubleshooting

- Missing `PICOVOICE_ACCESS_KEY`: diarization will fail. Add it to `.env`.
- `ffmpeg` not found: install it and ensure it is on `PATH`.
- Missing spaCy model: masking falls back to regex-only.

