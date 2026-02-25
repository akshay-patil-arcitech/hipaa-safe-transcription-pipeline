import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import whisperx_utils as utils
from whisperx_pipeline import process_media_whisperx
from whisper_word_ts_pipeline import process_media_whisper_with_word_ts

_FFMPEG_BIN = r"C:\ffmpeg-8.0.1-full_build-shared\ffmpeg-8.0.1-full_build-shared\bin"
if os.path.isdir(_FFMPEG_BIN):
    current_path = os.environ.get("PATH", "")
    if _FFMPEG_BIN not in current_path:
        os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + current_path

# Force Pyannote to use soundfile backend to avoid torchcodec dependency on Windows.
os.environ.setdefault("PYANNOTE_AUDIO_BACKEND", "soundfile")

app = FastAPI(title="WhisperX Pipeline API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    hf_token: str | None = Form(None),
    model: str = Form("medium"),
    force_diarization: bool = Form(True),
    speaker_roles: str = Form("DOCTOR,INTERPRETER,PATIENT"),
    masking: bool = Form(True),
    masking_entities: str | None = Form(None),
    hipaa: bool = Form(False),
    beep_audio: bool = Form(False),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix or ".wav"
    stem = Path(file.filename).stem
    temp_fd, temp_path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    os.close(temp_fd)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        roles = [item.strip() for item in speaker_roles.split(",") if item.strip()]
        force_three = force_diarization and len(roles) == 3

        effective_entities = masking_entities
        if hipaa and not effective_entities:
            effective_entities = "PHI"

        segments, output_path, beeped_path = process_media_whisperx(
            media_path=temp_path,
            model_name=model,
            device="auto",
            language=None,
            hf_token=hf_token,
            preset="quality",
            compute_type=None,
            threads=None,
            batch_size=None,
            chunk_size=30,
            num_workers=0,
            vad_onset=0.5,
            vad_offset=0.363,
            beam_size=None,
            best_of=None,
            temperatures=None,
            condition_on_previous_text=True,
            suppress_numerals=False,
            num_speakers=3 if force_three else None,
            diarize_min_speakers=3 if force_three else None,
            diarize_max_speakers=3 if force_three else None,
            fill_nearest_speaker=True,
            merge_gap=0.2,
            word_merge_gap=0.2,
            min_segment_len=0.6,
            max_speakers=3,
            min_speaker_total=5.0,
            speaker_roles=roles or ["DOCTOR", "INTERPRETER", "PATIENT"],
            masking=masking or hipaa,
            masking_entities=utils._parse_masking_entities(effective_entities) or utils._parse_masking_entities("PHI"),
            masking_blocklist=[],
            masking_context_rules=True,
            per_segment_language=True,
            language_allowlist=["en", "es", "fr", "ar", "vi", "zh", "ht", "yue", "cmn"],
            beep_audio=beep_audio,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    payload = {"segments": segments, "output_file": output_path.name}
    if beeped_path:
        payload["beeped_audio_file"] = Path(beeped_path).name
    return payload


@app.post("/transcribe_whisper_word_ts")
def transcribe_whisper_word_ts(
    file: UploadFile = File(...),
    model: str = Form("medium"),
    falcon_key: str | None = Form(None),
    device: str = Form("auto"),
    speaker_roles: str = Form("DOCTOR,INTERPRETER,PATIENT"),
    masking: bool = Form(True),
    beep_audio: bool = Form(False),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix or ".wav"
    stem = Path(file.filename).stem
    temp_fd, temp_path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    os.close(temp_fd)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        roles = [item.strip() for item in speaker_roles.split(",") if item.strip()]
        segments, output_path, beeped_path = process_media_whisper_with_word_ts(
            media_path=temp_path,
            model_name=model,
            falcon_access_key=falcon_key,
            device=device,
            decode_options=None,
            threads=None,
            merge_gap=0.4,
            min_seg=0.6,
            collapse_short=True,
            pad_sec=0.1,
            max_speakers=3 if roles else None,
            min_speaker_total=None,
            masking=masking,
            masking_entities=utils._parse_masking_entities("PHI"),
            masking_blocklist=[],
            masking_context_rules=True,
            speaker_roles=roles or ["DOCTOR", "INTERPRETER", "PATIENT"],
            beep_audio=beep_audio,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    payload = {"segments": segments, "output_file": output_path.name}
    if beeped_path:
        payload["beeped_audio_file"] = Path(beeped_path).name
    return payload
