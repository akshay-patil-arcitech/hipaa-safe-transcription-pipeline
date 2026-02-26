import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from urllib.parse import urlparse
from urllib.request import urlretrieve
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
import whisperx
import pvfalcon
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline
from fastapi import FastAPI, File, Form, UploadFile

SegmentLike = namedtuple("SegmentLike", ["start_sec", "end_sec", "speaker_tag"])

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

DEFAULT_MASKING_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "DATE_TIME",
    "LOCATION",
    "IP_ADDRESS",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "BANK_ACCOUNT",
    "US_ITIN",
    "URL",
    "MEDICAL_LICENSE",
]
PII_ENTITY_SET = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "BANK_ACCOUNT",
    "US_ITIN",
    "URL",
]
PHI_ENTITY_SET = [
    "PERSON",
    "DATE_TIME",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_SSN_COMPACT",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "BANK_ACCOUNT",
    "US_ITIN",
    "URL",
    "MEDICAL_LICENSE",
    "DATE_OF_BIRTH",
    "MRN",
    "US_NPI",
    "ICD10",
    "CPT",
    "DATE_GENERIC",
    "AGE",
    "ADDRESS",
    "ZIP",
    "STATE_ID",
]

_ENTITY_TAG_MAP = {
    "PERSON": "[PERSON_NAME]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "US_SSN": "[SSN]",
    "US_SSN_COMPACT": "[SSN]",
    "US_PASSPORT": "[PASSPORT]",
    "US_DRIVER_LICENSE": "[DRIVER_LICENSE]",
    "US_ITIN": "[ITIN]",
    "IP_ADDRESS": "[IP]",
    "URL": "[URL]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IBAN_CODE": "[IBAN]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT]",
    "DATE_TIME": "[DATE]",
    "LOCATION": "[LOCATION]",
    "MEDICAL_LICENSE": "[MED_LICENSE]",
    "DATE_OF_BIRTH": "[DOB]",
    "MRN": "[MRN]",
    "US_NPI": "[NPI]",
    "ICD10": "[ICD10]",
    "CPT": "[CPT]",
    "DATE_GENERIC": "[DATE]",
    "AGE": "[AGE]",
    "ADDRESS": "[ADDRESS]",
    "ZIP": "[ZIP]",
    "STATE_ID": "[STATE_ID]",
}

_REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "US_SSN_COMPACT": re.compile(r"\b\d{9}\b"),
    "US_PASSPORT": re.compile(r"\b(?:passport|pp)\b[:\s#-]*[0-9A-Z]{6,9}\b", re.IGNORECASE),
    "US_DRIVER_LICENSE": re.compile(r"\b(?:dl|driver'?s license|driver license|lic(?:ense)?)\b[:\s#-]*[A-Z0-9]{5,15}\b", re.IGNORECASE),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "BANK_ACCOUNT": re.compile(r"\b\d{8,17}\b"),
    "US_ITIN": re.compile(r"\b9\d{2}-?\d{2}-?\d{4}\b"),
    "URL": re.compile(r"\bhttps?://[^\s]+\b"),
    "DATE_OF_BIRTH": re.compile(r"\b(?:dob|date of birth)\b[:\s-]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    "MRN": re.compile(r"\b(?:mrn|medical record|record #|patient id)\b[:\s-]*[A-Z0-9-]{6,15}\b", re.IGNORECASE),
    "US_NPI": re.compile(r"\b\d{10}\b"),
    "ICD10": re.compile(r"\b[A-TV-Z][0-9][0-9AB]\.?\w{0,4}\b"),
    "CPT": re.compile(r"\b\d{5}\b"),
    "DATE_GENERIC": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?)\b",
        re.IGNORECASE,
    ),
    "AGE": re.compile(r"\b(?:age|aged)\s*\d{1,3}\b", re.IGNORECASE),
    "ADDRESS": re.compile(
        r"\b\d{1,6}\s+[A-Za-z0-9\s.-]+(?:st|street|ave|avenue|rd|road|blvd|lane|ln|dr|drive|ct|court)\b",
        re.IGNORECASE,
    ),
    "ZIP": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "STATE_ID": re.compile(r"\b[A-Z]{2}\s*\d{4,10}\b"),
}

_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]|\[\[[A-Z][A-Z0-9_]*\]\]")
_PRESIDIO_CACHE: dict[str, tuple[object, object] | None] = {}
_SENTIMENT_PIPELINE = None
_SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_SPACY_MODEL_MAP = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
    "ru": "ru_core_news_sm",
    "uk": "uk_core_news_sm",
    "pl": "pl_core_news_sm",
    "ro": "ro_core_news_sm",
    "ca": "ca_core_news_sm",
    "el": "el_core_news_sm",
    "lt": "lt_core_news_sm",
    "ar": "ar_core_news_sm",
    "vi": "vi_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
}

_PHI_KEYWORD_MAP = {
    "name": "[PERSON_NAME]",
    "patient": "[PATIENT]",
    "dob": "[DOB]",
    "date of birth": "[DOB]",
    "mrn": "[MRN]",
    "medical record": "[MRN]",
    "address": "[ADDRESS]",
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "ssn": "[SSN]",
    "passport": "[PASSPORT]",
    "insurance": "[INSURANCE]",
    # Spanish
    "nombre": "[PERSON_NAME]",
    "paciente": "[PATIENT]",
    "fecha de nacimiento": "[DOB]",
    "historia clínica": "[MRN]",
    "dirección": "[ADDRESS]",
    "teléfono": "[PHONE]",
    "correo": "[EMAIL]",
    "seguro": "[INSURANCE]",
    # French
    "nom": "[PERSON_NAME]",
    "date de naissance": "[DOB]",
    "dossier médical": "[MRN]",
    "adresse": "[ADDRESS]",
    "téléphone": "[PHONE]",
    "courriel": "[EMAIL]",
    "assurance": "[INSURANCE]",
}


def _resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_seconds = int(round(seconds))
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _speaker_label(tag: int, roles: list[str] | None) -> str:
    if roles and 0 <= tag < len(roles):
        return roles[tag]
    if 0 <= tag < 26:
        return chr(ord("A") + tag)
    return str(tag)


def _duration(seg: SegmentLike) -> float:
    return seg.end_sec - seg.start_sec


def _merge_adjacent_same_speaker(segments: list[SegmentLike], merge_gap: float) -> list[SegmentLike]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker_tag == prev.speaker_tag and (seg.start_sec - prev.end_sec) <= merge_gap:
            merged[-1] = SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        else:
            merged.append(seg)
    return merged


def postprocess_segments(segments: list[SegmentLike], merge_gap: float, min_seg: float, collapse_short: bool) -> list[SegmentLike]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    merged = []
    for seg in segments:
        if _duration(seg) <= 0:
            continue
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg.start_sec - prev.end_sec
        if seg.speaker_tag == prev.speaker_tag and gap <= merge_gap:
            merged[-1] = SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        else:
            merged.append(seg)
    if not collapse_short or min_seg <= 0:
        return merged
    result = []
    i = 0
    while i < len(merged):
        seg = merged[i]
        if _duration(seg) >= min_seg:
            result.append(seg)
            i += 1
            continue
        prev = result[-1] if result else None
        nxt = merged[i + 1] if i + 1 < len(merged) else None
        if prev:
            result[-1] = SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        elif nxt:
            merged[i + 1] = SegmentLike(min(seg.start_sec, nxt.start_sec), nxt.end_sec, nxt.speaker_tag)
        else:
            result.append(seg)
        i += 1
    return result


def reassign_minor_speakers(
    segments: list[SegmentLike],
    max_speakers: int | None,
    min_speakers: int | None,
    min_total_sec: float | None,
    merge_gap: float,
) -> list[SegmentLike]:
    if not segments:
        return []
    if max_speakers is None and (min_total_sec is None or min_total_sec <= 0) and (min_speakers is None or min_speakers <= 0):
        return segments
    totals: dict[int, float] = {}
    for seg in segments:
        totals[seg.speaker_tag] = totals.get(seg.speaker_tag, 0.0) + _duration(seg)
    keep: set[int] = set()
    if max_speakers is not None and max_speakers > 0:
        top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:max_speakers]
        keep.update(tag for tag, _ in top)
    if min_total_sec is not None and min_total_sec > 0:
        for tag, total in totals.items():
            if total >= min_total_sec:
                keep.add(tag)
    if min_speakers is not None and min_speakers > 0:
        ranked = [tag for tag, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)]
        for tag in ranked:
            if len(keep) >= min_speakers:
                break
            keep.add(tag)
    if not keep:
        return segments
    sorted_segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    reassigned = []
    for i, seg in enumerate(sorted_segments):
        if seg.speaker_tag in keep:
            reassigned.append(seg)
            continue
        prev_keep = next((sorted_segments[j] for j in range(i - 1, -1, -1) if sorted_segments[j].speaker_tag in keep), None)
        next_keep = next((sorted_segments[j] for j in range(i + 1, len(sorted_segments)) if sorted_segments[j].speaker_tag in keep), None)
        if prev_keep and next_keep:
            gap_prev = abs(seg.start_sec - prev_keep.end_sec)
            gap_next = abs(next_keep.start_sec - seg.end_sec)
            target = prev_keep if gap_prev <= gap_next else next_keep
            reassigned.append(SegmentLike(seg.start_sec, seg.end_sec, target.speaker_tag))
        elif prev_keep:
            reassigned.append(SegmentLike(seg.start_sec, seg.end_sec, prev_keep.speaker_tag))
        elif next_keep:
            reassigned.append(SegmentLike(seg.start_sec, seg.end_sec, next_keep.speaker_tag))
        else:
            reassigned.append(seg)
    return _merge_adjacent_same_speaker(reassigned, merge_gap=merge_gap)


def apply_speaker_roles(segments: list[SegmentLike], roles: list[str]) -> list[SegmentLike]:
    if not segments or not roles:
        return segments
    totals: dict[int, float] = {}
    for seg in segments:
        totals[seg.speaker_tag] = totals.get(seg.speaker_tag, 0.0) + _duration(seg)
    ranked = [tag for tag, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)]
    role_map: dict[int, int] = {}
    for i, tag in enumerate(ranked):
        if i >= len(roles):
            break
        role_map[tag] = i
    remapped = []
    for seg in segments:
        if seg.speaker_tag in role_map:
            remapped.append(SegmentLike(seg.start_sec, seg.end_sec, role_map[seg.speaker_tag]))
        else:
            remapped.append(seg)
    return remapped


def is_wav_16k_mono(path: str) -> bool:
    try:
        with wave.open(path, "rb") as audio:
            return (
                audio.getframerate() == TARGET_SAMPLE_RATE
                and audio.getnchannels() == TARGET_CHANNELS
                and audio.getsampwidth() == TARGET_SAMPLE_WIDTH
            )
    except wave.Error:
        return False


def _create_temp_wav_path(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    return path


def _safe_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def convert_media_to_wav(input_path: str, output_path: str):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH. Install ffmpeg to convert media to WAV.")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@contextmanager
def prepared_audio(media_path: str):
    if is_wav_16k_mono(media_path):
        yield media_path
        return
    temp_audio_path = _create_temp_wav_path(prefix="whisper_word_ts_")
    convert_media_to_wav(media_path, temp_audio_path)
    try:
        yield temp_audio_path
    finally:
        _safe_remove(temp_audio_path)


def _load_wave_as_float32(audio_path: str) -> tuple[np.ndarray, int]:
    with wave.open(audio_path, "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        if sample_rate != TARGET_SAMPLE_RATE or channels != TARGET_CHANNELS or width != TARGET_SAMPLE_WIDTH:
            raise ValueError("Audio must be 16kHz mono 16-bit PCM.")
        frames = audio.readframes(audio.getnframes())
    audio_i16 = np.frombuffer(frames, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32, sample_rate


def _slice_audio(audio_f32: np.ndarray, sample_rate: int, segment: SegmentLike, pad_sec: float):
    start = max(0.0, segment.start_sec - pad_sec)
    end = max(start, segment.end_sec + pad_sec)
    start_frame = max(0, int(start * sample_rate))
    end_frame = min(audio_f32.shape[0], int(end * sample_rate))
    if end_frame <= start_frame:
        return None
    return audio_f32[start_frame:end_frame]


def _parse_masking_entities(items: list[str] | None) -> list[str]:
    if not items:
        return DEFAULT_MASKING_ENTITIES
    expanded: list[str] = []
    for item in items:
        key = item.upper()
        if key == "PII":
            expanded.extend(PII_ENTITY_SET)
        elif key == "PHI" or key == "HIPAA":
            expanded.extend(PHI_ENTITY_SET)
        else:
            expanded.append(item)
    seen = set()
    deduped: list[str] = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _get_presidio(language: str | None):
    lang = (language or "en").lower()
    cached = _PRESIDIO_CACHE.get(lang)
    if cached is not None:
        return cached
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        _PRESIDIO_CACHE[lang] = None
        return None
    model_name = _SPACY_MODEL_MAP.get(lang)
    if not model_name:
        _PRESIDIO_CACHE[lang] = None
        return None
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": model_name}],
        "ner_model_configuration": {
            "labels_to_ignore": ["PERSON_NAME", "ID_N", "FAC", "LANGUAGE", "MISC", "ORDINAL", "CARDINAL"],
        },
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[lang])
        anonymizer = AnonymizerEngine()
        _PRESIDIO_CACHE[lang] = (analyzer, anonymizer)
        return analyzer, anonymizer
    except Exception:
        _PRESIDIO_CACHE[lang] = None
        return None


def _regex_mask(text: str, entities: list[str]) -> str:
    masked = text
    for entity in entities:
        pattern = _REGEX_PATTERNS.get(entity)
        if pattern:
            tag = _ENTITY_TAG_MAP.get(entity, "[REDACTED]")
            masked = pattern.sub(tag, masked)
    return masked


def _apply_phi_keywords(text: str) -> str:
    if not text:
        return text
    masked = text
    for key, tag in _PHI_KEYWORD_MAP.items():
        masked = re.sub(rf"\b{re.escape(key)}\b", tag, masked, flags=re.IGNORECASE)
    return masked


def _mask_text(text: str, entities: list[str], language: str | None) -> str:
    text = _apply_phi_keywords(text)
    engine = _get_presidio(language)
    if engine is None:
        return _regex_mask(text, entities)
    analyzer, anonymizer = engine
    lang = (language or "en").lower()
    try:
        supported = set(analyzer.get_supported_entities(language=lang))
        entities = [e for e in entities if e in supported] or entities
    except Exception:
        pass
    try:
        results = analyzer.analyze(text=text, entities=entities, language=lang)
    except Exception:
        return _regex_mask(text, entities)
    if not results:
        return _regex_mask(text, entities)
    from presidio_anonymizer.entities import OperatorConfig
    operators = {
        entity: OperatorConfig("replace", {"new_value": _ENTITY_TAG_MAP.get(entity, "[REDACTED]")})
        for entity in entities
    }
    operators["DEFAULT"] = OperatorConfig("replace", {"new_value": "[REDACTED]"})
    masked = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
    return _regex_mask(masked, entities)

def _get_sentiment_pipeline(device: str):
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is not None:
        return _SENTIMENT_PIPELINE
    tokenizer = AutoTokenizer.from_pretrained(_SENTIMENT_MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(_SENTIMENT_MODEL_NAME)
    device_id = 0 if device == "cuda" else -1
    _SENTIMENT_PIPELINE = hf_pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        top_k=None,
    )
    return _SENTIMENT_PIPELINE

def _predict_sentiment(text: str, device: str) -> str:
    if not text:
        return "neutral"
    pipe = _get_sentiment_pipeline(device)
    outputs = pipe(text)
    # outputs is list[list[dict]] when top_k=None
    if not outputs or not outputs[0]:
        return "neutral"
    best = max(outputs[0], key=lambda x: x.get("score", 0.0))
    label = str(best.get("label", "")).lower()
    # normalize model labels to simple sentiment names
    if "negative" in label:
        return "negative"
    if "positive" in label:
        return "positive"
    return "neutral"


def _align_words_for_segment(text: str, lang: str | None, seg_audio: np.ndarray, seg_start: float, device: str, cache: dict):
    if not text:
        return []
    seg_duration = len(seg_audio) / TARGET_SAMPLE_RATE
    segments = [{"text": text, "start": 0.0, "end": seg_duration}]
    key = (lang or "en").lower()
    cached = cache.get(key)
    if cached:
        align_model, metadata = cached
    else:
        align_model, metadata = whisperx.load_align_model(language_code=key, device=device)
        cache[key] = (align_model, metadata)
    aligned = whisperx.align(segments, align_model, metadata, seg_audio, device)
    word_segments = aligned.get("word_segments", [])
    words = []
    for w in word_segments:
        word = (w.get("word") or "").strip()
        if not word:
            continue
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue
        words.append({"word": word, "start": float(start) + seg_start, "end": float(end) + seg_start})
    return words


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
    intervals = []
    for seg in transcriptions:
        for word in seg.get("words") or []:
            token = str(word.get("word", ""))
            if _PLACEHOLDER_RE.search(token):
                start = float(word.get("start") or 0.0)
                end = float(word.get("end") or 0.0)
                if end > start:
                    intervals.append((start, end))
    return _merge_intervals(intervals)


def _apply_beeps(audio: np.ndarray, intervals: list[tuple[float, float]], sample_rate: int, freq_hz: float, gain: float) -> np.ndarray:
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


def run(
    media_path: str,
    model_name: str = "medium",
    falcon_access_key: str | None = None,
    device: str = "auto",
    speaker_roles: list[str] | None = None,
    masking: bool = True,
    masking_entities: list[str] | None = None,
    sentiment: bool = True,
    include_word_ts: bool = True,
    max_speakers: int | None = None,
    min_speakers: int | None = None,
    min_speaker_sec: float | None = None,
    diar_merge_gap: float = 0.4,
    diar_min_seg: float = 0.6,
    diar_collapse_short: bool = True,
    force_roles: bool = False,
    beep_audio: bool = False,
    beep_freq_hz: float = 1000.0,
    beep_gain: float = 0.4,
):
    media_path = str(Path(media_path).resolve())
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Input file not found: {media_path}")

    device = _resolve_device(device)
    model = whisper.load_model(model_name, device=device)
    access_key = falcon_access_key or os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise ValueError("Picovoice AccessKey is required. Set PICOVOICE_ACCESS_KEY or use --falcon-key.")
    diarizer = pvfalcon.create(access_key=access_key)

    output_path = Path.cwd() / f"{Path(media_path).stem}_transcribes_word_ts.json"
    align_cache: dict = {}

    with prepared_audio(media_path) as audio_path:
        raw_segments = diarizer.process_file(audio_path)
        segments = [SegmentLike(s.start_sec, s.end_sec, s.speaker_tag) for s in raw_segments]
        if max_speakers is not None and min_speakers is not None and max_speakers < min_speakers:
            raise ValueError("max_speakers must be >= min_speakers.")
        if force_roles and speaker_roles:
            if max_speakers is None:
                max_speakers = len(speaker_roles)
            if min_speakers is None:
                min_speakers = len(speaker_roles)

        segments = postprocess_segments(segments, merge_gap=diar_merge_gap, min_seg=diar_min_seg, collapse_short=diar_collapse_short)
        segments = reassign_minor_speakers(
            segments,
            max_speakers=max_speakers if max_speakers and max_speakers > 0 else None,
            min_speakers=min_speakers if min_speakers and min_speakers > 0 else None,
            min_total_sec=min_speaker_sec,
            merge_gap=diar_merge_gap,
        )
        segments = _merge_adjacent_same_speaker(segments, merge_gap=diar_merge_gap)
        if speaker_roles:
            segments = apply_speaker_roles(segments, speaker_roles)

        transcriptions = []
        need_word_ts = include_word_ts or beep_audio
        audio_f32, sample_rate = _load_wave_as_float32(audio_path)

        for segment in segments:
            segment_audio = _slice_audio(audio_f32, sample_rate, segment, pad_sec=0.1)
            if segment_audio is None or segment_audio.size == 0:
                continue
            result = model.transcribe(segment_audio)
            text = (result.get("text") or "").strip()
            lang = result.get("language")
            if not text:
                continue
            if masking:
                entities = _parse_masking_entities(masking_entities)
                text = _mask_text(text, entities, lang)
            sentiment_label = _predict_sentiment(text, device) if sentiment else None
            words = []
            if need_word_ts:
                words = _align_words_for_segment(text, lang, segment_audio, segment.start_sec, device, align_cache)
            item = {
                "speaker": _speaker_label(segment.speaker_tag, speaker_roles),
                "transcribe": text,
                "start_ts": _format_ts(segment.start_sec),
                "end_ts": _format_ts(segment.end_sec),
            }
            if need_word_ts:
                item["words"] = words
            if sentiment_label is not None:
                item["sentiment"] = sentiment_label
            transcriptions.append(item)

        beeped_path = None
        if beep_audio:
            intervals = _collect_beep_intervals(transcriptions)
            beeped = _apply_beeps(audio_f32, intervals, sample_rate, beep_freq_hz, beep_gain)
            beeped_path = Path.cwd() / f"{Path(media_path).stem}_beeped.wav"
            sf.write(beeped_path, beeped, sample_rate)

    diarizer.delete()

    output_segments = transcriptions
    if not include_word_ts:
        output_segments = []
        for seg in transcriptions:
            if "words" in seg:
                seg = dict(seg)
                seg.pop("words", None)
            output_segments.append(seg)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_segments, file, ensure_ascii=False, indent=2)

    result = {"segments": output_segments, "output_file": output_path.name}
    if beeped_path:
        result["beeped_audio_file"] = beeped_path.name
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone Whisper word-level timestamp pipeline with optional beeping."
    )
    parser.add_argument("--input", required=True, help="Path to input media (audio or video).")
    parser.add_argument("--model", default="medium", help="Whisper model name.")
    parser.add_argument("--falcon-key", default=None, help="Picovoice AccessKey for Falcon diarization.")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, or cuda.")
    parser.add_argument("--speaker-roles", default="DOCTOR,INTERPRETER,PATIENT")
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--masking-entities", default="PHI")
    parser.add_argument("--beep-audio", action="store_true")
    parser.add_argument("--beep-freq-hz", type=float, default=1000.0)
    parser.add_argument("--beep-gain", type=float, default=0.4)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--min-speaker-sec", type=float, default=None)
    parser.add_argument("--diar-merge-gap", type=float, default=0.4)
    parser.add_argument("--diar-min-seg", type=float, default=0.6)
    parser.add_argument("--no-diar-collapse-short", action="store_true")
    parser.add_argument("--force-roles", action="store_true")
    args = parser.parse_args()

    roles = [r.strip() for r in args.speaker_roles.split(",") if r.strip()]
    entities = [e.strip().upper() for e in args.masking_entities.split(",") if e.strip()] if args.masking_entities else None
    output = run(
        media_path=args.input,
        model_name=args.model,
        falcon_access_key=args.falcon_key,
        device=args.device,
        speaker_roles=roles if roles else None,
        masking=args.masking,
        masking_entities=entities,
        beep_audio=args.beep_audio,
        beep_freq_hz=args.beep_freq_hz,
        beep_gain=args.beep_gain,
        max_speakers=args.max_speakers,
        min_speakers=args.min_speakers,
        min_speaker_sec=args.min_speaker_sec,
        diar_merge_gap=args.diar_merge_gap,
        diar_min_seg=args.diar_min_seg,
        diar_collapse_short=not args.no_diar_collapse_short,
        force_roles=args.force_roles,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


app = FastAPI()


def _download_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported.")
    ext = Path(parsed.path).suffix or ".mp4"
    fd, path = tempfile.mkstemp(prefix="whisper_word_ts_url_", suffix=ext)
    os.close(fd)
    urlretrieve(url, path)
    return path


@app.post("/transcribe_word_ts")
async def transcribe_word_ts_api(
    file: UploadFile | None = File(default=None),
    public_url: str | None = Form(default=None),
    model_name: str = Form(default="medium"),
    falcon_access_key: str | None = Form(default=None),
    device: str = Form(default="auto"),
    speaker_roles: str | None = Form(default="DOCTOR,INTERPRETER,PATIENT"),
    masking: bool = Form(default=True),
    masking_entities: str | None = Form(default="PHI"),
    sentiment: bool = Form(default=True),
    word_to_word: bool = Form(default=True),
    max_speakers: int | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    min_speaker_sec: float | None = Form(default=None),
    diar_merge_gap: float = Form(default=0.4),
    diar_min_seg: float = Form(default=0.6),
    diar_collapse_short: bool = Form(default=True),
    force_roles: bool = Form(default=False),
    beep_audio: bool = Form(default=False),
    beep_freq_hz: float = Form(default=1000.0),
    beep_gain: float = Form(default=0.4),
):
    if not file and not public_url:
        raise ValueError("Provide either a file upload or public_url.")

    temp_path = None
    try:
        if file is not None:
            suffix = Path(file.filename or "").suffix or ".mp4"
            fd, temp_path = tempfile.mkstemp(prefix="whisper_word_ts_upload_", suffix=suffix)
            os.close(fd)
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            media_path = temp_path
        else:
            media_path = _download_public_url(public_url)  # type: ignore[arg-type]

        roles = [r.strip() for r in (speaker_roles or "").split(",") if r.strip()] or None
        entities = [e.strip().upper() for e in (masking_entities or "").split(",") if e.strip()] or None
        result = run(
            media_path=media_path,
            model_name=model_name,
            falcon_access_key=falcon_access_key,
            device=device,
            speaker_roles=roles,
            masking=masking,
            masking_entities=entities,
            sentiment=sentiment,
            include_word_ts=word_to_word,
            max_speakers=max_speakers,
            min_speakers=min_speakers,
            min_speaker_sec=min_speaker_sec,
            diar_merge_gap=diar_merge_gap,
            diar_min_seg=diar_min_seg,
            diar_collapse_short=diar_collapse_short,
            force_roles=force_roles,
            beep_audio=beep_audio,
            beep_freq_hz=beep_freq_hz,
            beep_gain=beep_gain,
        )
        return result
    finally:
        if temp_path:
            _safe_remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("word_ts_pipeline_standalone:app", host="0.0.0.0", port=8000, reload=False)
