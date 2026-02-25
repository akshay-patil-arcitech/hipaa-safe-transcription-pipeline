import os
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import main as pipeline

app = FastAPI(title="Live Analysis Pipeline API", version="1.0.0")

_model_cache: dict[tuple[str, str], tuple[object, str]] = {}
_cache_lock = threading.Lock()


def _get_cached_model(model_name: str, device: str | None):
    resolved_device = pipeline._resolve_device(device)
    key = (model_name, resolved_device)
    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached
    model, resolved_device = pipeline.load_whisper_model(model_name, device=resolved_device)
    with _cache_lock:
        _model_cache[key] = (model, resolved_device)
    return model, resolved_device


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    model: str = Form("small"),
    device: str = Form("auto"),
    falcon_key: str | None = Form(None),
    threads: int | None = Form(None),
    language: str | None = Form(None),
    beam_size: int | None = Form(None),
    best_of: int | None = Form(None),
    temperature: float | None = Form(None),
    no_previous_text: bool = Form(False),
    fast: bool = Form(False),
    masking: bool = Form(False),
    masking_entities: str | None = Form(None),
    masking_blocklist: str | None = Form(None),
    masking_context_rules: bool = Form(True),
    max_speakers: int | None = Form(None),
    min_speaker_total: float | None = Form(None),
    speaker_roles: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    pipeline.set_torch_threads(threads)
    model_obj, resolved_device = _get_cached_model(model, device=device)

    args = SimpleNamespace(
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        no_previous_text=no_previous_text,
        fast=fast,
    )
    decode_options = pipeline._build_decode_options(args, resolved_device)

    access_key = pipeline._get_falcon_access_key(falcon_key)
    diarizer = pipeline.pvfalcon.create(access_key=access_key)

    suffix = Path(file.filename).suffix or ".wav"
    stem = Path(file.filename).stem
    output_path = Path.cwd() / f"{stem}_transcripts.json"
    temp_fd, temp_path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    os.close(temp_fd)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with pipeline.prepared_audio(temp_path) as audio_path:
            result = pipeline.process_audio(
                audio_path,
                model_obj,
                diarizer,
                decode_options=decode_options,
                output_path=output_path,
                merge_gap=0.4,
                min_seg=0.6,
                collapse_short=True,
                pad_sec=0.1,
                max_speakers=max_speakers,
                min_speaker_total=min_speaker_total,
                masking=masking,
                masking_entities=pipeline._parse_masking_entities(masking_entities),
                masking_blocklist=pipeline._parse_blocklist(masking_blocklist),
                masking_context_rules=masking_context_rules,
                speaker_roles=[item.strip() for item in speaker_roles.split(",")] if speaker_roles else None,
            )
    finally:
        diarizer.delete()
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return {"segments": result, "output_file": output_path.name}
