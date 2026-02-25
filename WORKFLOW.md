# WhisperX Workflow (Developer Level)

This document explains the current WhisperX-based workflow used by `whisperx_api.py`, including diarization, transcription, multilingual handling, masking, and word-level alignment. It is written as an implementation-oriented guide.

**Entry Point**

Run the API:
```bash
uvicorn whisperx_api:app --reload
```

The API exposes two pipelines:
1. `/transcribe` uses WhisperX diarization + Whisper transcription + WhisperX alignment.
2. `/transcribe_whisper_word_ts` uses Whisper diarization + Whisper transcription + WhisperX alignment.

The primary entry is `/transcribe`.

**Files and Responsibilities**

1. `whisperx_api.py`
   - FastAPI app.
   - Loads `.env` to fetch `PICOVOICE_ACCESS_KEY` and optional `HF_TOKEN`.
   - Ensures FFmpeg is discoverable on PATH.
   - Sets `PYANNOTE_AUDIO_BACKEND=soundfile` to avoid TorchCodec issues.
   - Routes user requests to the correct pipeline function.

2. `whisperx_pipeline.py`
   - WhisperX pipeline: diarization + Whisper transcription + WhisperX alignment for word timestamps.
   - Multilingual detection and per-segment language decision.
   - Speaker role mapping and post-processing heuristics.
   - Masking pipeline for PHI/PII/HIPAA with Presidio + regex fallback.

3. `whisperx_utils.py`
   - Masking utilities, Presidio integration, and aggressive regex-based PHI/PII detection.
   - Language-specific tagging and suppression of false positives.

4. `whisper_word_ts_pipeline.py`
   - Whisper diarization + Whisper transcription + WhisperX alignment.
   - Uses Whisper for language detection and transcription logic, WhisperX only for word alignment.

5. `main.py`
   - Original Whisper-only pipeline used as baseline for logic and decoding behavior.

**High-Level Flow for `/transcribe`**

1. Request parsing and defaults.
   - File upload and parameter parsing.
   - Default model is `medium` unless overridden.
   - `force_diarization` defaults to `true` and uses diarization model.
   - `speaker_roles` defaults to `DOCTOR,INTERPRETER,PATIENT`.
   - `masking` and `masking_entities` control PHI/PII/HIPAA masking.

2. Environment and configuration.
   - `.env` is loaded for `PICOVOICE_ACCESS_KEY`.
   - Optional `HF_TOKEN` is used for gated Hugging Face diarization models if provided.
   - `PYANNOTE_AUDIO_BACKEND` is set to `soundfile`.

3. Audio preparation.
   - Input media is converted to 16kHz, mono, 16-bit PCM WAV if needed.
   - Temporary WAV files are created and cleaned up automatically.

4. Diarization (WhisperX).
   - `whisperx.DiarizationPipeline` is loaded with the selected device.
   - `min_speakers` and `max_speakers` can be forced to stabilize multi-speaker sessions.
   - Raw diarization segments are produced with start/end timestamps and speaker labels.

5. Diarization post-processing.
   - Segments are merged when gaps are small and speaker labels match.
   - Short segments are optionally collapsed into nearest neighbors.
   - Minor speakers can be reassigned to the closest major speaker.
   - Additional heuristics split Q/A boundaries and handle trailing short interjections.

6. Transcription (Whisper).
   - A Whisper model is loaded once and reused for all segments.
   - Each diarized segment is sliced from the WAV with a small padding window.
   - Per-segment language detection is performed unless `--language` is forced.
   - Language allowlist can be set to reduce false detections.
   - Decoding options follow Whisper behavior: beam size, best-of, temperature, `fp16`.

7. Word-level alignment (WhisperX).
   - WhisperX alignment is run per segment using the detected language.
   - Output includes `words` array with `word`, `start`, `end` timestamps.
   - Alignment only uses WhisperX, transcription remains Whisper-based.

8. Masking (PHI/PII/HIPAA).
   - Presidio NER is used when available for the segment language.
   - Regex fallback is used for structured identifiers and missing languages.
   - Aggressive coverage includes PERSON_NAME, DOB, MRN, NPI, SSN, IDs, addresses, dates, and more.
   - A context-driven keyword pass applies additional masking for HIPAA-sensitive phrases.
   - A final regex sweep removes residual patterns.

9. Speaker role assignment.
   - Roles are mapped by total speaking time by default.
   - Optional heuristics adjust roles based on content cues.

10. Output persistence.
   - Results are written to `<input_stem>_whisperx_transcripts.json`.
   - Each segment contains timestamps, speaker label, transcription, and word-level alignment.

**High-Level Flow for `/transcribe_whisper_word_ts`**

This pipeline is identical in output format but differs in diarization:
1. Diarization uses Picovoice Falcon (`pvfalcon`).
2. Transcription uses Whisper.
3. Alignment uses WhisperX only for word timestamps.

This matches Whisper transcription logic more closely, but depends on Falcon diarization quality.

**Output Schema**

Each segment object:
```json
{
  "speaker": "DOCTOR",
  "transcribe": "Hello there.",
  "start_ts": "0:00:08.416",
  "end_ts": "0:00:11.712",
  "words": [
    { "word": "Hello", "start": 8.416, "end": 8.76 },
    { "word": "there.", "start": 8.78, "end": 9.02 }
  ]
}
```

Output file name:
- WhisperX: `<input_stem>_whisperx_transcripts.json`
- Whisper+alignment: `<input_stem>_transcribes_word_ts.json`

**Masking Configuration**

Parameters in `/transcribe`:
1. `masking` boolean enables masking.
2. `masking_entities` accepts `PII`, `PHI`, `HIPAA`, or explicit entity names.
3. `hipaa` boolean forces an aggressive masking set.

Masking uses Presidio if installed, with a multilingual fallback.
Regex patterns are used to cover structured identifiers in all languages.

**Language Handling**

Default behavior:
1. Whisper detects language per segment.
2. Optionally restrict to a language allowlist.
3. Alignment uses the detected language for correct token alignment.

This supports multilingual audio in a single file.

**Performance Notes**

1. `medium` is a balance between quality and speed.
2. GPU improves inference and alignment significantly.
3. Smaller models increase speed but reduce accuracy.
4. Diarization quality is the dominant accuracy factor for speaker attribution.

**Error Handling**

Common failures:
1. Missing `PICOVOICE_ACCESS_KEY` for Falcon diarization.
2. Missing HF token for gated diarization models.
3. Missing FFmpeg on PATH.
4. Unsupported languages for Presidio or missing spaCy models.

Errors are raised as standard Python exceptions and logged.

**How to Debug**

1. Check `diarization_transcription.log` for segment-by-segment events.
2. Verify detected language per segment.
3. Review raw diarization segments before post-processing.
4. Compare masked output with unmasked to validate regex/NER behavior.

**API Payloads**

`/transcribe` expects a multipart form with:
1. `file` (audio/video file).
2. `model` optional, default `medium`.
3. `force_diarization` optional, boolean.
4. `speaker_roles` optional, CSV string.
5. `masking` optional, boolean.
6. `masking_entities` optional, `PII,PHI,HIPAA` or explicit entity list.
7. `hipaa` optional, boolean.

`/transcribe_whisper_word_ts` expects:
1. `file`.
2. `model` optional.
3. `falcon_key` optional.
4. `device` optional.
5. `speaker_roles` optional.
6. `masking` optional.

**Key Implementation Decisions**

1. Whisper is used for transcription to preserve Whisper language detection behavior.
2. WhisperX is used only for word-level alignment to avoid translation bias.
3. Masking is aggressive by design, preferring false positives over missed PHI.
4. Diarization is post-processed to reduce turn-taking errors and split Q/A properly.
