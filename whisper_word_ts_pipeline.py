import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import whisper
import whisperx

import main as legacy


def _format_ts(seconds: float) -> str:
    return legacy._format_ts(seconds)


def _speaker_label(tag: int) -> str:
    return legacy._speaker_label(tag)


def _load_align_model_cached(lang: str, device: str, cache: dict):
    key = (lang or "en").lower()
    cached = cache.get(key)
    if cached:
        return cached
    align_model, metadata = whisperx.load_align_model(language_code=key, device=device)
    cache[key] = (align_model, metadata)
    return align_model, metadata


def _align_words_for_segment(
    text: str,
    lang: str | None,
    seg_audio,
    seg_start: float,
    device: str,
    cache: dict,
):
    if not text:
        return []
    seg_duration = len(seg_audio) / legacy.TARGET_SAMPLE_RATE
    segments = [{"text": text, "start": 0.0, "end": seg_duration}]
    align_model, metadata = _load_align_model_cached(lang or "en", device, cache)
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
        words.append(
            {
                "word": word,
                "start": float(start) + seg_start,
                "end": float(end) + seg_start,
            }
        )
    return words


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


def process_media_whisper_with_word_ts(
    media_path: str,
    model_name: str = "medium",
    falcon_access_key: str | None = None,
    device: str | None = None,
    decode_options: dict | None = None,
    threads: int | None = None,
    merge_gap: float = 0.4,
    min_seg: float = 0.6,
    collapse_short: bool = True,
    pad_sec: float = 0.1,
    max_speakers: int | None = None,
    min_speaker_total: float | None = None,
    masking: bool = False,
    masking_entities: list[str] | None = None,
    masking_blocklist: list[str] | None = None,
    masking_context_rules: bool = True,
    speaker_roles: list[str] | None = None,
    beep_audio: bool = False,
    beep_freq_hz: float = 1000.0,
    beep_gain: float = 0.4,
    beep_output_path: str | None = None,
):
    media_path = str(Path(media_path).resolve())
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Input file not found: {media_path}")

    legacy.set_torch_threads(threads)
    model, resolved_device = legacy.load_whisper_model(model_name, device=device)
    access_key = legacy._get_falcon_access_key(falcon_access_key)
    diarizer = legacy.pvfalcon.create(access_key=access_key)
    output_path = Path.cwd() / f"{Path(media_path).stem}_transcribes_word_ts.json"
    if decode_options is None:
        decode_options = {"fp16": resolved_device != "cpu"}

    legacy.log_event(f"Started processing media: {media_path}")
    align_cache: dict = {}
    try:
        with legacy.prepared_audio(media_path) as audio_path:
            raw_segments = diarizer.process_file(audio_path)
            segments = [
                legacy.SegmentLike(s.start_sec, s.end_sec, s.speaker_tag)
                for s in raw_segments
            ]
            segments = legacy.postprocess_segments(
                segments=segments,
                merge_gap=merge_gap,
                min_seg=min_seg,
                collapse_short=collapse_short,
            )
            segments = legacy.reassign_minor_speakers(
                segments=segments,
                max_speakers=max_speakers,
                min_total_sec=min_speaker_total,
                merge_gap=merge_gap,
            )
            if speaker_roles:
                segments = legacy.apply_speaker_roles(segments, speaker_roles)

            transcriptions = []
            audio_f32, sample_rate = legacy._load_wave_as_float32(audio_path)

            for segment in segments:
                segment_audio = legacy._slice_audio(audio_f32, sample_rate, segment, pad_sec=pad_sec)
                if segment_audio is None or segment_audio.size == 0:
                    legacy.log_event("Skipped empty or invalid segment.")
                    continue

                transcription, lang = legacy.transcribe_audio_segment(segment_audio, model, decode_options)
                if masking:
                    entities = masking_entities or legacy.DEFAULT_MASKING_ENTITIES
                    transcription = legacy.mask_text(
                        transcription,
                        entities,
                        lang,
                        blocklist=masking_blocklist,
                        context_rules=masking_context_rules,
                    )

                if speaker_roles and 0 <= segment.speaker_tag < len(speaker_roles):
                    speaker_label = speaker_roles[segment.speaker_tag]
                else:
                    speaker_label = _speaker_label(segment.speaker_tag)

                word_level = _align_words_for_segment(
                    transcription,
                    lang,
                    segment_audio,
                    segment.start_sec,
                    resolved_device,
                    align_cache,
                )

                transcriptions.append(
                    {
                        "speaker": speaker_label,
                        "transcribe": transcription,
                        "start_ts": _format_ts(segment.start_sec),
                        "end_ts": _format_ts(segment.end_sec),
                        "words": word_level,
                    }
                )
                if output_path is not None:
                    legacy._write_json(output_path, transcriptions)
    finally:
        diarizer.delete()

    beeped_path = None
    if beep_audio:
        intervals = _collect_beep_intervals(transcriptions)
        beeped = _apply_beeps(
            audio=audio_f32,
            intervals=intervals,
            sample_rate=sample_rate,
            freq_hz=beep_freq_hz,
            gain=beep_gain,
        )
        if beep_output_path:
            beeped_path = Path(beep_output_path)
        else:
            beeped_path = Path.cwd() / f"{Path(media_path).stem}_beeped.wav"
        sf.write(beeped_path, beeped, sample_rate)

    legacy.log_event(f"Completed processing media: {media_path}")
    return transcriptions, output_path, beeped_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Whisper transcription + WhisperX alignment for word-level timestamps."
    )
    parser.add_argument("--input", required=True, help="Path to input media (audio or video).")
    parser.add_argument("--model", default="medium", help="Whisper model name (default: medium).")
    parser.add_argument("--falcon-key", default=None, help="Picovoice AccessKey for Falcon.")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, or cuda.")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--merge-gap", type=float, default=0.4)
    parser.add_argument("--min-seg", type=float, default=0.6)
    parser.add_argument("--no-collapse-short", action="store_true")
    parser.add_argument("--pad-sec", type=float, default=0.1)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--min-speaker-total", type=float, default=None)
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--masking-entities", default=None)
    parser.add_argument("--masking-blocklist", default=None)
    parser.add_argument("--masking-context-rules", action="store_true")
    parser.add_argument("--speaker-roles", default=None)
    parser.add_argument("--beep-audio", action="store_true")
    parser.add_argument("--beep-freq-hz", type=float, default=1000.0)
    parser.add_argument("--beep-gain", type=float, default=0.4)
    parser.add_argument("--beep-output-path", default=None)
    args = parser.parse_args()

    decode_options = legacy._build_decode_options(args, legacy._resolve_device(args.device))

    result, output_path, beeped_path = process_media_whisper_with_word_ts(
        media_path=args.input,
        model_name=args.model,
        falcon_access_key=args.falcon_key,
        device=args.device,
        decode_options=decode_options,
        threads=args.threads,
        merge_gap=args.merge_gap,
        min_seg=args.min_seg,
        collapse_short=not args.no_collapse_short,
        pad_sec=args.pad_sec,
        max_speakers=args.max_speakers,
        min_speaker_total=args.min_speaker_total,
        masking=args.masking,
        masking_entities=legacy._parse_masking_entities(args.masking_entities),
        masking_blocklist=legacy._parse_blocklist(args.masking_blocklist),
        masking_context_rules=args.masking_context_rules,
        speaker_roles=[item.strip() for item in args.speaker_roles.split(",")] if args.speaker_roles else None,
        beep_audio=args.beep_audio,
        beep_freq_hz=args.beep_freq_hz,
        beep_gain=args.beep_gain,
        beep_output_path=args.beep_output_path,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Output: {output_path.name}")
    if beeped_path:
        print(f"Beeped audio: {beeped_path.name}")
