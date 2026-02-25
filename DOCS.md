# Project Documentation

## Overview
This project provides a **fully local pipeline** for:
- Multilingual transcription (English/Spanish focus, but supports Whisper multilingual)
- Speaker diarization (Doctor / Interpreter / Patient)
- Redaction (PII/PHI masking)
- Word‑level timestamps
- Optional audio beeping for redacted words
- Optional sentiment per segment

It is designed for medical conversations and interpreter workflows, with strong emphasis on HIPAA‑style redaction.

## Tech Stack
- **Transcription**: OpenAI Whisper (`openai-whisper`)
- **Word alignment**: WhisperX alignment
- **Diarization**: Picovoice Falcon (`pvfalcon`)
- **Masking**: Presidio + regex fallbacks
- **Audio handling**: `ffmpeg`, `soundfile`, `numpy`
- **API**: FastAPI (optional)
- **Sentiment**: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (optional)
- **Python**: 3.12+

## High‑Level Flow
1. **Input**
   - Accepts local audio/video, or public URL (FastAPI).
2. **Audio normalization**
   - Convert to 16kHz mono PCM WAV.
3. **Diarization (Falcon)**
   - Segment audio by speaker and post‑process segments.
4. **Transcription (Whisper)**
   - Transcribe each segment, detect language.
5. **Redaction**
   - Presidio + regex coverage for PII/PHI fields.
6. **Word alignment (WhisperX)**
   - Align words to timestamps.
7. **Optional beep**
   - Replace audio regions for redacted words with tone.
8. **Output**
   - JSON with segments + words + sentiment (optional) + beeped audio path.

## Output Format (Example)
```json
{
  "speaker": "INTERPRETER",
  "transcribe": "[PERSON_NAME], mucho gusto. ...",
  "sentiment": "neutral",
  "start_ts": "0:00:36.192",
  "end_ts": "0:00:50.336",
  "words": [
    {"word": "[PERSON_NAME],", "start": 36.1920, "end": 38.2980},
    {"word": "mucho", "start": 38.3190, "end": 38.4190}
  ]
}
```

## Accuracy & Evaluation Methodology
Accuracy was measured using three independent methods:

1. **Gemini YouTube comparison**
   - Uploaded Whisper transcripts to Gemini.
   - Provided the YouTube video link.
   - Asked Gemini to compare transcript vs. video.
   - Reported overall pipeline accuracy: **~92–95%** for the English‑Spanish sample.

2. **NoteGPT comparison**
   - Retrieved YouTube transcript from NoteGPT.
   - Compared NoteGPT transcript with our transcript inside Gemini.
   - Reported accuracy: **~98%** for the same video.

3. **Kaggle audio + transcript datasets**
   - Compared against provided ground‑truth transcripts.
   - Verified with manual checks + Gemini + ChatGPT.
   - Overall accuracy: **~95%** (approximate).

> These results are based on limited datasets and may vary by audio quality, language mix, speaker overlap, and model size.

## Whisper Model Benchmarks (Local Laptop no GPU)
Measured on a bilingual English‑Spanish YouTube sample:

- **Whisper tiny (39M / 75MB)**  
  - Time: **~2:30**  
  - Accuracy: **75–80%**

- **Whisper base (74M / 142MB)**  
  - Time: **~4:00**  
  - Accuracy: **80–85%**

- **Whisper small (244M / 466MB)**  
  - Time: **~13–15 min**  
  - Accuracy: **80–85%**

- **Whisper medium (769M / 1.5GB)**  
  - Time: **~30–35 min**  
  - Accuracy: **92–95%**

Other language checks:

- **Whisper medium (Arabic‑English)**  
  - Time: **~20–25 min**  
  - Accuracy: **~90%**

- **Whisper base (French‑English)**  
  - Time: **~3 min**  
  - Accuracy: **~60%**

- **Whisper small (French‑English)**  
  - Time: **~10–11 min**  
  - Accuracy: **French ~75% / English ~100%**

## Notes & Limitations
- Accuracy depends heavily on audio quality, speaker overlap, and microphone placement.
- Diarization improves with clear turn‑taking and clean audio.
- Redaction is conservative (aggressive masking) to maximize PHI/PII coverage.
- Sentiment is optional and currently limited to **negative/neutral/positive**.
