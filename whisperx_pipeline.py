import json
import os
import re
from pathlib import Path
import numpy as np
import soundfile as sf

import whisper
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.audio import SAMPLE_RATE

import whisperx_utils as utils


def _format_ts(seconds: float) -> str:
    return utils._format_ts(seconds)


def _speaker_label(tag: int, roles: list[str] | None):
    if roles and 0 <= tag < len(roles):
        return roles[tag]
    return utils._speaker_label(tag)


def _assign_int_speakers(speaker_str: str, mapping: dict[str, int]) -> int:
    if speaker_str not in mapping:
        mapping[speaker_str] = len(mapping)
    return mapping[speaker_str]


def _group_words(words: list[dict], max_gap_sec: float) -> list[dict]:
    grouped = []
    current = None
    for w in words:
        if "speaker" not in w or w.get("start") is None or w.get("end") is None:
            continue
        speaker = w["speaker"]
        text = w.get("word", "").strip()
        if not text:
            continue
        if current and current["speaker"] == speaker and w["start"] <= current["end"] + max_gap_sec:
            current["text"] = (current["text"] + " " + text).strip()
            current["end"] = max(current["end"], w["end"])
        else:
            current = {"speaker": speaker, "text": text, "start": w["start"], "end": w["end"]}
            grouped.append(current)
    return grouped


def _resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        try:
            import torch
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _detect_language(model, audio: "np.ndarray", fallback: str = "en") -> str:
    try:
        return model.detect_language(audio)
    except Exception:
        return fallback


def _detect_language_with_allowlist(
    model,
    audio: "np.ndarray",
    allowlist: set[str] | None,
    fallback: str,
) -> str:
    if allowlist is None:
        return _detect_language(model, audio, fallback=fallback)
    try:
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        _, probs = model.detect_language(mel)
        best_lang = None
        best_prob = -1.0
        for lang, prob in probs.items():
            if lang in allowlist and prob > best_prob:
                best_lang = lang
                best_prob = prob
        if best_lang:
            return best_lang
    except Exception:
        pass
    return fallback


def _normalize_transcript(text: str) -> str:
    if not text:
        return text
    replacements = {
        r"\blanguage world\b": "LanguageLine",
        r"\bclodidad\b": "Claudia",
    }
    updated = text
    for pattern, repl in replacements.items():
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)
    return updated


def _apply_label_heuristics(transcriptions: list[dict]) -> list[dict]:
    prev_speaker = None
    for segment in transcriptions:
        text = segment.get("transcribe", "")
        lowered = text.lower().replace("\u00ed", "i")
        interpreter_cues = (
            "language line",
            "language world",
            "services",
            "confidential",
            "short sentences",
            "interpreting",
            "interpreter",
            "por confidencialidad",
            "confidencial",
            "parece que",
            "ultrasonido",
            "¿verdad",
            "verdad",
            "muy bien",
            "por favor",
            "hablen",
            "frases cortas",
        )
        doctor_cues = (
            "i am the doctor",
            "i'm the doctor",
            "soy la doctora",
            "soy el doctor",
            "yo soy la doctora",
            "yo soy el doctor",
            "i speak a little spanish",
            "hablo un poquito",
            "necesito usar un interprete",
            "necesito usar un interpreter",
            "medical history",
            "ultrasound",
            "gestational",
            "pregnancy",
            "protocol",
            "yo soy el medico",
            "soy el medico",
            "soy la medica",
        )
        patient_cues = (
            "si, si",
            "sí, sí",
            "si.",
            "sí.",
            "okay",
            "ok",
        )
        if any(cue in lowered for cue in interpreter_cues):
            segment["speaker"] = "INTERPRETER"
        elif any(cue in lowered for cue in doctor_cues):
            segment["speaker"] = "DOCTOR"
        elif any(cue in lowered for cue in patient_cues):
            segment["speaker"] = "PATIENT"
        if lowered.strip() in ("si, si", "si, si.", "si si"):
            if prev_speaker == "DOCTOR":
                segment["speaker"] = "PATIENT"
        prev_speaker = segment.get("speaker")
    return transcriptions


def _split_turn_taking(transcriptions: list[dict]) -> list[dict]:
    if not transcriptions:
        return transcriptions
    split_markers = (
        "hello",
        "hi",
        "hola",
        "buenos",
        "buenas",
    )
    updated: list[dict] = []
    for seg in transcriptions:
        text = seg.get("transcribe", "")
        if not text:
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in split_markers) and "." in text:
            parts = [p.strip() for p in re.split(r"[.!?]\s+", text) if p.strip()]
            if len(parts) > 1:
                # Keep first sentence in current segment, split the rest
                first = parts[0]
                rest = parts[1:]
                seg_copy = dict(seg)
                seg_copy["transcribe"] = first
                updated.append(seg_copy)
                for chunk in rest:
                    new_seg = dict(seg)
                    new_seg["transcribe"] = chunk
                    updated.append(new_seg)
                continue
        updated.append(seg)
    return updated


def _split_question_answer_segments(transcriptions: list[dict]) -> list[dict]:
    if not transcriptions:
        return transcriptions
    answer_cues = (
        "i'm well",
        "i am well",
        "very well",
        "muy bien",
        "bien",
    )
    result: list[dict] = []
    for seg in transcriptions:
        text = seg.get("transcribe", "")
        words = seg.get("words") or []
        if not text or not words:
            result.append(seg)
            continue
        lowered = text.lower()
        if "?" in text and any(cue in lowered for cue in answer_cues):
            # split at first question mark word
            split_idx = None
            for i, w in enumerate(words):
                if "?" in str(w.get("word", "")):
                    split_idx = i
                    break
            if split_idx is not None and split_idx + 1 < len(words):
                first_words = words[: split_idx + 1]
                second_words = words[split_idx + 1 :]
                first_text = " ".join(w.get("word", "").strip() for w in first_words).strip()
                second_text = " ".join(w.get("word", "").strip() for w in second_words).strip()
                if first_text and second_text:
                    first_seg = dict(seg)
                    second_seg = dict(seg)
                    first_seg["transcribe"] = first_text
                    second_seg["transcribe"] = second_text
                    first_seg["end_ts"] = _format_ts(first_words[-1]["end"])
                    second_seg["start_ts"] = _format_ts(second_words[0]["start"])
                    # assume response is interpreter if answer cue present
                    second_seg["speaker"] = "INTERPRETER"
                    second_seg["words"] = second_words
                    first_seg["words"] = first_words
                    result.append(first_seg)
                    result.append(second_seg)
                    continue
        result.append(seg)
    return result


def _split_trailing_si_si(transcriptions: list[dict]) -> list[dict]:
    if not transcriptions:
        return transcriptions
    result: list[dict] = []
    for seg in transcriptions:
        text = (seg.get("transcribe") or "").strip()
        words = seg.get("words") or []
        lowered = text.lower()
        if ("sí" in lowered or "si" in lowered) and words:
            # find trailing "sí, sí" tokens
            idx = None
            for i, w in enumerate(words):
                if str(w.get("word", "")).strip().lower().startswith(("sí", "si")):
                    idx = i
                    break
            if idx is not None and idx > 0:
                first_words = words[:idx]
                second_words = words[idx:]
                first_text = " ".join(w.get("word", "").strip() for w in first_words).strip()
                second_text = " ".join(w.get("word", "").strip() for w in second_words).strip()
                if first_text and second_text:
                    first_seg = dict(seg)
                    second_seg = dict(seg)
                    first_seg["transcribe"] = first_text
                    second_seg["transcribe"] = second_text
                    first_seg["end_ts"] = _format_ts(first_words[-1]["end"])
                    second_seg["start_ts"] = _format_ts(second_words[0]["start"])
                    second_seg["speaker"] = "PATIENT"
                    second_seg["words"] = second_words
                    first_seg["words"] = first_words
                    result.append(first_seg)
                    result.append(second_seg)
                    continue
        result.append(seg)
    return result


def _dedupe_repeated_phrases(transcriptions: list[dict], window_sec: float = 4.0) -> list[dict]:
    if not transcriptions:
        return transcriptions
    cleaned: list[dict] = []
    last_seen: dict[str, float] = {}
    for seg in transcriptions:
        text = (seg.get("transcribe") or "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not key:
            continue
        start_ts = seg.get("start_ts", "0:00:00.000")
        try:
            h, m, s = start_ts.split(":")
            start_sec = int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            start_sec = 0.0
        last = last_seen.get(key)
        if last is not None and (start_sec - last) <= window_sec:
            # drop tight repeats (e.g., echo loops)
            continue
        last_seen[key] = start_sec
        cleaned.append(seg)
    return cleaned


def _smooth_short_segments(
    segments: list[utils.SegmentLike],
    min_len: float,
    merge_gap: float,
) -> list[utils.SegmentLike]:
    if not segments or min_len <= 0:
        return segments
    segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    smoothed: list[utils.SegmentLike] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if utils._duration(seg) >= min_len:
            smoothed.append(seg)
            i += 1
            continue
        prev = smoothed[-1] if smoothed else None
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if prev and (seg.start_sec - prev.end_sec) <= merge_gap:
            smoothed[-1] = utils.SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        elif nxt and (nxt.start_sec - seg.end_sec) <= merge_gap:
            segments[i + 1] = utils.SegmentLike(min(seg.start_sec, nxt.start_sec), nxt.end_sec, nxt.speaker_tag)
        elif prev:
            smoothed[-1] = utils.SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        elif nxt:
            segments[i + 1] = utils.SegmentLike(min(seg.start_sec, nxt.start_sec), nxt.end_sec, nxt.speaker_tag)
        else:
            smoothed.append(seg)
        i += 1
    return smoothed


def _resolve_compute_type(device: str, compute_type: str | None, preset: str) -> str:
    if compute_type:
        return compute_type
    if device == "cuda":
        return "float16"
    if preset == "speed":
        return "int8"
    return "float32"


def _default_threads() -> int:
    count = os.cpu_count() or 4
    return max(1, min(8, count // 2))


def _build_asr_options(
    preset: str,
    beam_size: int | None,
    best_of: int | None,
    temperatures: list[float] | None,
    condition_on_previous_text: bool | None,
    suppress_numerals: bool,
) -> dict:
    preset = (preset or "balanced").lower()
    if preset == "speed":
        options = {
            "beam_size": 1,
            "best_of": 1,
            "temperatures": [0.0],
            "condition_on_previous_text": False,
            "compression_ratio_threshold": None,
            "log_prob_threshold": None,
            "no_speech_threshold": None,
        }
    elif preset == "quality":
        options = {
            "beam_size": 5,
            "best_of": 5,
            "temperatures": [0.0, 0.2, 0.4, 0.6],
            "condition_on_previous_text": True,
        }
    else:
        options = {
            "beam_size": 3,
            "best_of": 3,
            "temperatures": [0.0, 0.2, 0.4],
            "condition_on_previous_text": False,
        }

    if beam_size is not None:
        options["beam_size"] = beam_size
    if best_of is not None:
        options["best_of"] = best_of
    if temperatures:
        options["temperatures"] = temperatures
    if condition_on_previous_text is not None:
        options["condition_on_previous_text"] = condition_on_previous_text
    if suppress_numerals:
        options["suppress_numerals"] = True

    return options


_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]|\[\[[A-Z][A-Z0-9_]*\]\]")


def _merge_intervals(intervals: list[tuple[float, float]], gap: float = 0.01) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _collect_beep_intervals(transcriptions: list[dict]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for seg in transcriptions:
        for word in seg.get("words") or []:
            token = str(word.get("word", ""))
            if not token:
                continue
            if _PLACEHOLDER_RE.search(token):
                start = float(word.get("start") or 0.0)
                end = float(word.get("end") or 0.0)
                if end > start:
                    intervals.append((start, end))
    return _merge_intervals(intervals)


def _apply_beeps(
    audio: np.ndarray,
    intervals: list[tuple[float, float]],
    sample_rate: int,
    freq_hz: float,
    gain: float,
) -> np.ndarray:
    if not intervals:
        return audio
    output = np.array(audio, copy=True)
    for start_sec, end_sec in intervals:
        start_idx = max(0, int(start_sec * sample_rate))
        end_idx = min(output.shape[0], int(end_sec * sample_rate))
        if end_idx <= start_idx:
            continue
        t = np.arange(end_idx - start_idx, dtype=np.float32) / float(sample_rate)
        tone = np.sin(2.0 * np.pi * freq_hz * t) * float(gain)
        output[start_idx:end_idx] = tone
    return output


def process_media_whisperx(
    media_path: str,
    model_name: str = "medium",
    device: str = "auto",
    language: str | None = None,
    hf_token: str | None = None,
    preset: str = "balanced",
    compute_type: str | None = None,
    threads: int | None = None,
    batch_size: int | None = None,
    chunk_size: int = 30,
    num_workers: int = 0,
    vad_onset: float = 0.5,
    vad_offset: float = 0.363,
    beam_size: int | None = None,
    best_of: int | None = None,
    temperatures: list[float] | None = None,
    condition_on_previous_text: bool | None = None,
    suppress_numerals: bool = False,
    num_speakers: int | None = None,
    diarize_min_speakers: int | None = None,
    diarize_max_speakers: int | None = None,
    fill_nearest_speaker: bool = False,
    merge_gap: float = 0.4,
    word_merge_gap: float = 0.3,
    min_segment_len: float = 0.0,
    max_speakers: int | None = None,
    min_speaker_total: float | None = None,
    speaker_roles: list[str] | None = None,
    masking: bool = False,
    masking_entities: list[str] | None = None,
    masking_blocklist: list[str] | None = None,
    masking_context_rules: bool = True,
    per_segment_language: bool = False,
    language_allowlist: list[str] | None = None,
    beep_audio: bool = False,
    beep_freq_hz: float = 1000.0,
    beep_gain: float = 0.4,
    beep_output_path: str | None = None,
):
    media_path = str(Path(media_path).resolve())
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Input file not found: {media_path}")

    device = _resolve_device(device)
    compute_type = _resolve_compute_type(device, compute_type, preset)
    asr_options = _build_asr_options(
        preset=preset,
        beam_size=beam_size,
        best_of=best_of,
        temperatures=temperatures,
        condition_on_previous_text=condition_on_previous_text,
        suppress_numerals=suppress_numerals,
    )
    vad_options = {
        "chunk_size": chunk_size,
        "vad_onset": vad_onset,
        "vad_offset": vad_offset,
    }
    if threads is None:
        threads = _default_threads()
    # Use OpenAI Whisper for transcription, then WhisperX for alignment.
    audio = whisperx.load_audio(media_path)
    whisper_model = whisper.load_model(model_name, device=device)
    allowlist = set(language_allowlist) if language_allowlist else None
    global_lang = _detect_language_with_allowlist(
        whisper_model,
        audio[: SAMPLE_RATE * 30],
        allowlist=allowlist,
        fallback=language or "en",
    )
    result = whisper_model.transcribe(
        audio,
        language=language or global_lang,
        task="transcribe",
    )
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", "en"), device=device
    )
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    if hf_token:
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        if num_speakers is None and diarize_min_speakers is None and diarize_max_speakers is None:
            if speaker_roles and len(speaker_roles) == 3:
                diarize_min_speakers = 3
                diarize_max_speakers = 3
        diarize_segments = diarize_model(
            audio,
            num_speakers=num_speakers,
            min_speakers=diarize_min_speakers,
            max_speakers=diarize_max_speakers,
        )
        result = assign_word_speakers(
            diarize_segments,
            result,
            fill_nearest=fill_nearest_speaker,
        )
    else:
        # fallback: treat all words as one speaker
        for w in result.get("word_segments", []):
            w["speaker"] = "SPEAKER_00"

    words = result.get("word_segments", [])
    grouped = _group_words(words, max_gap_sec=word_merge_gap)

    speaker_map: dict[str, int] = {}
    segments = []
    for g in grouped:
        speaker_id = _assign_int_speakers(g["speaker"], speaker_map)
        segments.append(
            utils.SegmentLike(g["start"], g["end"], speaker_id)
        )

    segments = utils.reassign_minor_speakers(
        segments=segments,
        max_speakers=max_speakers,
        min_total_sec=min_speaker_total,
        merge_gap=merge_gap,
    )
    segments = utils._merge_adjacent_same_speaker(segments, merge_gap=merge_gap)
    segments = _smooth_short_segments(segments, min_len=max(min_segment_len, 0.6), merge_gap=merge_gap)
    segments = utils._merge_adjacent_same_speaker(segments, merge_gap=merge_gap)
    # Second pass: reassign any tiny speakers after smoothing
    segments = utils.reassign_minor_speakers(
        segments=segments,
        max_speakers=max_speakers,
        min_total_sec=min_speaker_total,
        merge_gap=merge_gap,
    )
    segments = utils._merge_adjacent_same_speaker(segments, merge_gap=merge_gap)
    if speaker_roles:
        segments = utils.apply_speaker_roles(segments, speaker_roles)

    transcriptions = []
    align_cache: dict[str, tuple[object, object]] = {}

    def _get_align_for_lang(lang: str):
        cached = align_cache.get(lang)
        if cached:
            return cached
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        align_cache[lang] = (align_model, metadata)
        return align_model, metadata

    for seg in segments:
        if min_segment_len > 0 and (seg.end_sec - seg.start_sec) < min_segment_len:
            continue
        seg_start = max(0.0, seg.start_sec)
        seg_end = max(seg_start, seg.end_sec)
        start_frame = int(seg_start * SAMPLE_RATE)
        end_frame = int(seg_end * SAMPLE_RATE)
        seg_audio = audio[start_frame:end_frame]
        if seg_audio is None or seg_audio.size == 0:
            continue

        if per_segment_language:
            try:
                # Use Whisper's language detection when segment is long enough
                if (seg_end - seg_start) >= 3.0:
                    lang_code = _detect_language_with_allowlist(
                        whisper_model,
                        seg_audio,
                        allowlist=allowlist,
                        fallback=global_lang,
                    )
                else:
                    lang_code = result.get("language", global_lang)
            except Exception:
                lang_code = result.get("language", global_lang)
            seg_result = whisper_model.transcribe(
                seg_audio,
                language=lang_code,
                task="transcribe",
            )
            try:
                align_model, metadata = _get_align_for_lang(lang_code or "en")
                aligned = whisperx.align(seg_result["segments"], align_model, metadata, seg_audio, device)
            except Exception:
                aligned = seg_result
            seg_words = aligned.get("word_segments", [])
            text = " ".join(s.get("text", "").strip() for s in aligned.get("segments", [])).strip()
            word_level = [
                {
                    "word": w.get("word", "").strip(),
                    "start": (w.get("start") or 0.0) + seg_start,
                    "end": (w.get("end") or 0.0) + seg_start,
                }
                for w in seg_words
                if w.get("start") is not None
                and w.get("end") is not None
                and str(w.get("word", "")).strip()
            ]
            lang_for_masking = lang_code
        else:
            # get words that fall inside this segment
            seg_words = [
                w for w in grouped
                if w["start"] >= seg.start_sec and w["end"] <= seg.end_sec
            ]
            word_level = [
                {
                    "word": w.get("word", "").strip(),
                    "start": w.get("start"),
                    "end": w.get("end"),
                }
                for w in words
                if w.get("start") is not None
                and w.get("end") is not None
                and w.get("start") >= seg.start_sec
                and w.get("end") <= seg.end_sec
                and str(w.get("word", "")).strip()
            ]
            text = " ".join(w["text"] for w in seg_words).strip()
            lang_for_masking = result.get("language")

        text = _normalize_transcript(text)
        if not text:
            continue
        if masking:
            entities = masking_entities or utils.DEFAULT_MASKING_ENTITIES
            text = utils.mask_text(
                text,
                entities,
                lang_for_masking,
                blocklist=masking_blocklist,
                context_rules=masking_context_rules,
            )
        transcriptions.append(
            {
                "speaker": _speaker_label(seg.speaker_tag, speaker_roles),
                "transcribe": text,
                "start_ts": _format_ts(seg.start_sec),
                "end_ts": _format_ts(seg.end_sec),
                "words": word_level,
            }
        )

    transcriptions = _split_turn_taking(transcriptions)
    transcriptions = _split_question_answer_segments(transcriptions)
    transcriptions = _split_trailing_si_si(transcriptions)
    transcriptions = _apply_label_heuristics(transcriptions)
    transcriptions = _dedupe_repeated_phrases(transcriptions)

    output_path = Path.cwd() / f"{Path(media_path).stem}_whisperx_transcripts.json"
    utils.write_json(output_path, transcriptions)
    beeped_path = None
    if beep_audio:
        intervals = _collect_beep_intervals(transcriptions)
        beeped = _apply_beeps(
            audio=audio,
            intervals=intervals,
            sample_rate=SAMPLE_RATE,
            freq_hz=beep_freq_hz,
            gain=beep_gain,
        )
        if beep_output_path:
            beeped_path = Path(beep_output_path)
        else:
            beeped_path = Path.cwd() / f"{Path(media_path).stem}_beeped.wav"
        sf.write(beeped_path, beeped, SAMPLE_RATE)
    return transcriptions, output_path, beeped_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WhisperX pipeline with alignment + optional diarization."
    )
    parser.add_argument("--input", required=True, help="Path to input media.")
    parser.add_argument("--model", default="medium", help="Whisper model name.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--language", default=None, help="Force language code (e.g., en).")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for diarization.")
    parser.add_argument("--preset", default="balanced", help="speed, balanced, or quality.")
    parser.add_argument("--compute-type", default=None, help="Override compute type (e.g., int8, float16, float32).")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for whisperx.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vad-onset", type=float, default=0.5)
    parser.add_argument("--vad-offset", type=float, default=0.363)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--temperatures", default=None, help="Comma-separated temperatures (e.g., 0.0,0.2,0.4).")
    parser.add_argument("--condition-on-previous-text", action="store_true")
    parser.add_argument("--suppress-numerals", action="store_true")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--diarize-min-speakers", type=int, default=None)
    parser.add_argument("--diarize-max-speakers", type=int, default=None)
    parser.add_argument("--fill-nearest-speaker", action="store_true")
    parser.add_argument("--merge-gap", type=float, default=0.4)
    parser.add_argument("--word-merge-gap", type=float, default=0.3)
    parser.add_argument("--min-segment-len", type=float, default=0.0)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--min-speaker-total", type=float, default=None)
    parser.add_argument("--speaker-roles", default=None)
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--masking-entities", default=None)
    parser.add_argument("--masking-blocklist", default=None)
    parser.add_argument("--masking-context-rules", action="store_true")
    parser.add_argument("--beep-audio", action="store_true")
    parser.add_argument("--beep-freq-hz", type=float, default=1000.0)
    parser.add_argument("--beep-gain", type=float, default=0.4)
    parser.add_argument("--beep-output-path", default=None)
    args = parser.parse_args()

    transcripts, output_path, beeped_path = process_media_whisperx(
        media_path=args.input,
        model_name=args.model,
        device=args.device,
        language=args.language,
        hf_token=args.hf_token,
        preset=args.preset,
        compute_type=args.compute_type,
        threads=args.threads,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        vad_onset=args.vad_onset,
        vad_offset=args.vad_offset,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperatures=[float(t) for t in args.temperatures.split(",")] if args.temperatures else None,
        condition_on_previous_text=args.condition_on_previous_text,
        suppress_numerals=args.suppress_numerals,
        num_speakers=args.num_speakers,
        diarize_min_speakers=args.diarize_min_speakers,
        diarize_max_speakers=args.diarize_max_speakers,
        fill_nearest_speaker=args.fill_nearest_speaker,
        merge_gap=args.merge_gap,
        word_merge_gap=args.word_merge_gap,
        min_segment_len=args.min_segment_len,
        max_speakers=args.max_speakers,
        min_speaker_total=args.min_speaker_total,
        speaker_roles=[item.strip() for item in args.speaker_roles.split(",")] if args.speaker_roles else None,
        masking=args.masking,
        masking_entities=utils._parse_masking_entities(args.masking_entities),
        masking_blocklist=utils._parse_blocklist(args.masking_blocklist),
        masking_context_rules=args.masking_context_rules,
        beep_audio=args.beep_audio,
        beep_freq_hz=args.beep_freq_hz,
        beep_gain=args.beep_gain,
        beep_output_path=args.beep_output_path,
    )
    print(json.dumps(transcripts, ensure_ascii=False, indent=2))
    print(f"Output: {output_path.name}")
    if beeped_path:
        print(f"Beeped audio: {beeped_path.name}")
