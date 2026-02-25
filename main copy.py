import whisper
import pvfalcon
import wave
import os
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
import json
import numpy as np
import torch
from collections import namedtuple

# Setup logging for HIPAA compliance (track data access and processing)
logging.basicConfig(filename='diarization_transcription.log', level=logging.INFO)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit PCM
SegmentLike = namedtuple("SegmentLike", ["start_sec", "end_sec", "speaker_tag"])

def log_event(event):
    logging.info(f"Event: {event}")

def load_dotenv_file(path: str = ".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass

def _resolve_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def set_torch_threads(threads: int | None):
    if threads is None:
        return
    if threads < 1:
        return
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)

def load_whisper_model(model_name: str, device: str | None):
    # Choose 'small' for balance between size and accuracy
    resolved_device = _resolve_device(device)
    return whisper.load_model(model_name, device=resolved_device), resolved_device

def transcribe_audio_segment(segment_audio, model, decode_options: dict):
    # Transcribe audio segment with Whisper
    result = model.transcribe(segment_audio, **decode_options)
    log_event("Transcription complete for segment.")
    return result['text']

def _format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"

def _speaker_label(tag: int) -> str:
    if 0 <= tag < 26:
        return chr(ord("A") + tag)
    return str(tag)

def _write_json(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except OSError:
        pass

def is_wav_16k_mono(path: str) -> bool:
    try:
        with wave.open(path, 'rb') as audio:
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
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg to convert video/audio to WAV."
        )

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

    log_event(f"Converting media to WAV: {input_path}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to convert media. Exit code: {exc.returncode}"
        ) from exc

@contextmanager
def prepared_audio(media_path: str):
    temp_audio_path = None
    if is_wav_16k_mono(media_path):
        yield media_path
        return

    temp_audio_path = _create_temp_wav_path(prefix="falcon_whisper_input_")
    convert_media_to_wav(media_path, temp_audio_path)
    try:
        yield temp_audio_path
    finally:
        _safe_remove(temp_audio_path)

def _load_wave_as_float32(audio_path: str) -> tuple[np.ndarray, int]:
    with wave.open(audio_path, 'rb') as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        if sample_rate != TARGET_SAMPLE_RATE or channels != TARGET_CHANNELS or width != TARGET_SAMPLE_WIDTH:
            raise ValueError("Audio must be 16kHz mono 16-bit PCM at this stage.")
        frames = audio.readframes(audio.getnframes())
    audio_i16 = np.frombuffer(frames, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32, sample_rate

def _slice_audio(audio_f32: np.ndarray, sample_rate: int, segment, pad_sec: float):
    start = max(0.0, segment.start_sec - pad_sec)
    end = max(start, segment.end_sec + pad_sec)
    start_frame = max(0, int(start * sample_rate))
    end_frame = max(start_frame, int(end * sample_rate))
    end_frame = min(end_frame, audio_f32.shape[0])
    if end_frame <= start_frame:
        return None
    return audio_f32[start_frame:end_frame]

def _duration(segment: SegmentLike) -> float:
    return segment.end_sec - segment.start_sec

def _merge_segments(a: SegmentLike, b: SegmentLike) -> SegmentLike:
    return SegmentLike(
        start_sec=min(a.start_sec, b.start_sec),
        end_sec=max(a.end_sec, b.end_sec),
        speaker_tag=a.speaker_tag,
    )

def postprocess_segments(
    segments: list[SegmentLike],
    merge_gap: float,
    min_seg: float,
    collapse_short: bool,
) -> list[SegmentLike]:
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))

    merged: list[SegmentLike] = []
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

    result: list[SegmentLike] = []
    i = 0
    while i < len(merged):
        seg = merged[i]
        if _duration(seg) >= min_seg:
            result.append(seg)
            i += 1
            continue

        prev = result[-1] if result else None
        nxt = merged[i + 1] if i + 1 < len(merged) else None
        if prev and nxt:
            gap_prev = abs(seg.start_sec - prev.end_sec)
            gap_next = abs(nxt.start_sec - seg.end_sec)
            if gap_prev <= gap_next:
                result[-1] = _merge_segments(prev, seg)
            else:
                merged[i + 1] = SegmentLike(min(seg.start_sec, nxt.start_sec), nxt.end_sec, nxt.speaker_tag)
        elif prev:
            result[-1] = _merge_segments(prev, seg)
        elif nxt:
            merged[i + 1] = SegmentLike(min(seg.start_sec, nxt.start_sec), nxt.end_sec, nxt.speaker_tag)
        else:
            result.append(seg)
        i += 1

    return result

def _merge_adjacent_same_speaker(
    segments: list[SegmentLike],
    merge_gap: float,
) -> list[SegmentLike]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    merged: list[SegmentLike] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_sec - prev.end_sec
        if seg.speaker_tag == prev.speaker_tag and gap <= merge_gap:
            merged[-1] = SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        else:
            merged.append(seg)
    return merged

def reassign_minor_speakers(
    segments: list[SegmentLike],
    max_speakers: int | None,
    min_total_sec: float | None,
    merge_gap: float,
) -> list[SegmentLike]:
    if not segments:
        return []
    if max_speakers is None and (min_total_sec is None or min_total_sec <= 0):
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

    if not keep:
        return segments

    sorted_segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    reassigned: list[SegmentLike] = []
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

def process_audio(
    audio_path,
    model,
    diarizer,
    decode_options: dict,
    output_path: Path | None = None,
    merge_gap: float = 0.4,
    min_seg: float = 0.6,
    collapse_short: bool = True,
    pad_sec: float = 0.1,
    max_speakers: int | None = None,
    min_speaker_total: float | None = None,
):
    # Diarize audio with Falcon
    log_event(f"Started diarizing audio: {audio_path}")
    raw_segments = diarizer.process_file(audio_path)
    segments = [
        SegmentLike(s.start_sec, s.end_sec, s.speaker_tag)
        for s in raw_segments
    ]
    segments = postprocess_segments(
        segments=segments,
        merge_gap=merge_gap,
        min_seg=min_seg,
        collapse_short=collapse_short,
    )
    segments = reassign_minor_speakers(
        segments=segments,
        max_speakers=max_speakers,
        min_total_sec=min_speaker_total,
        merge_gap=merge_gap,
    )
    
    transcriptions = []
    audio_f32, sample_rate = _load_wave_as_float32(audio_path)
    
    for segment in segments:
        # Extract audio for the current speaker segment
        segment_audio = _slice_audio(audio_f32, sample_rate, segment, pad_sec=pad_sec)
        if segment_audio is None or segment_audio.size == 0:
            log_event("Skipped empty or invalid segment.")
            continue
        
        # Transcribe the segment using Whisper
        transcription = transcribe_audio_segment(segment_audio, model, decode_options)
        
        # Store speaker and transcription info
        transcriptions.append({
            "speaker": _speaker_label(segment.speaker_tag),
            "transcribe": transcription,
            "start_ts": _format_ts(segment.start_sec),
            "end_ts": _format_ts(segment.end_sec),
        })
        if output_path is not None:
            _write_json(output_path, transcriptions)
    
    return transcriptions

def _get_falcon_access_key(cli_access_key: str | None) -> str:
    load_dotenv_file()
    access_key = cli_access_key or os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise ValueError(
            "Falcon requires a Picovoice AccessKey. Provide --falcon-key "
            "or set PICOVOICE_ACCESS_KEY."
        )
    return access_key

def _build_decode_options(args, device: str) -> dict:
    options: dict = {}
    if args.language:
        options["language"] = args.language
    if args.beam_size is not None:
        options["beam_size"] = args.beam_size
    if args.best_of is not None:
        options["best_of"] = args.best_of
    if args.temperature is not None:
        options["temperature"] = args.temperature
    options["condition_on_previous_text"] = not args.no_previous_text

    if args.fast:
        options.setdefault("beam_size", 1)
        options.setdefault("best_of", 1)
        options["condition_on_previous_text"] = False
        options["compression_ratio_threshold"] = None
        options["logprob_threshold"] = None
        options["no_speech_threshold"] = None

    options["fp16"] = device != "cpu"
    return options

def process_media(
    media_path,
    model_name="small",
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
):
    media_path = str(Path(media_path).resolve())
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Input file not found: {media_path}")

    set_torch_threads(threads)
    model, resolved_device = load_whisper_model(model_name, device=device)
    access_key = _get_falcon_access_key(falcon_access_key)
    diarizer = pvfalcon.create(access_key=access_key)
    output_path = Path.cwd() / f"{Path(media_path).stem}_transcribes.json"
    if decode_options is None:
        decode_options = {"fp16": resolved_device != "cpu"}

    log_event(f"Started processing media: {media_path}")
    try:
        with prepared_audio(media_path) as audio_path:
            result = process_audio(
                audio_path,
                model,
                diarizer,
                decode_options=decode_options,
                output_path=output_path,
                merge_gap=merge_gap,
                min_seg=min_seg,
                collapse_short=collapse_short,
                pad_sec=pad_sec,
                max_speakers=max_speakers,
                min_speaker_total=min_speaker_total,
            )
    finally:
        diarizer.delete()
    log_event(f"Completed processing media: {media_path}")

    return result

def ensure_hipaa_compliance(audio_path, output):
    """
    Ensure HIPAA compliance during processing:
    1. Encrypt sensitive data.
    2. Log actions for auditing purposes.
    3. Delete sensitive data after use.
    """
    # Encrypt the files if needed (this part depends on how you manage encryption in your infrastructure)
    # Example: You may use a custom encryption function here, or rely on encrypted file storage.
    log_event(f"Processing audio file: {audio_path}")
    
    # Encrypting or protecting sensitive data can be done here.
    # Remember to delete or overwrite sensitive data once processing is complete.

    # After processing, ensure sensitive data is deleted or appropriately stored.
    log_event(f"Completed processing for audio: {audio_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diarize speakers with Falcon and transcribe with Whisper."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input media (audio or video).",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name (default: small).",
    )
    parser.add_argument(
        "--falcon-key",
        default=None,
        help="Picovoice AccessKey for Falcon (or set PICOVOICE_ACCESS_KEY).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Whisper device: auto, cpu, or cuda (default: auto).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads for torch (default: let torch decide).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g., en, es) to skip detection.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size (default: whisper default).",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Best-of samples (default: whisper default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Decoding temperature (default: whisper default).",
    )
    parser.add_argument(
        "--no-previous-text",
        action="store_true",
        help="Disable conditioning on previous text (faster, less context).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fastest decoding settings (lower accuracy).",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.4,
        help="Merge same-speaker segments with gaps <= this many seconds (default: 0.4).",
    )
    parser.add_argument(
        "--min-seg",
        type=float,
        default=0.6,
        help="Minimum segment length in seconds before collapsing (default: 0.6).",
    )
    parser.add_argument(
        "--no-collapse-short",
        action="store_true",
        help="Disable collapsing very short segments into neighbors.",
    )
    parser.add_argument(
        "--pad-sec",
        type=float,
        default=0.1,
        help="Pad segment audio on both sides in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Reassign minor speakers to the closest major speaker, keeping only this many (post-process).",
    )
    parser.add_argument(
        "--min-speaker-total",
        type=float,
        default=None,
        help="Keep speakers with at least this many total seconds (post-process).",
    )
    args = parser.parse_args()

    decode_options = _build_decode_options(args, _resolve_device(args.device))

    # Run the process
    result = process_media(
        args.input,
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
    )

    # Output the results: Speaker segments and their transcriptions
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for r in result:
        log_event(f"Speaker {r['speaker']} transcription: {r['transcribe']}")

