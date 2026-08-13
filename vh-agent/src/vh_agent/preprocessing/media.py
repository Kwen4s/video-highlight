import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from ..models import FrameSample, VideoInfo


class MediaError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MediaError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise MediaError(message) from exc


def probe_video(path: Path, language: str | None = None) -> VideoInfo:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"Video is not readable: {path}")

    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"No video stream found: {path}")

    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    numerator, denominator = (float(part) for part in rate.split("/"))
    fps = numerator / denominator if denominator else 0.0
    duration = float(payload.get("format", {}).get("duration") or video.get("duration") or 0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise MediaError(f"Invalid video metadata: {path}")
    return VideoInfo(
        path=path,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        title=_clean_title(path.stem),
        language=language,
    )


def extract_audio(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _valid_wav(output_path):
        return output_path
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return output_path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    frame_width: int,
) -> list[FrameSample]:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".complete"
    cache_key = json.dumps({"sample_fps": sample_fps, "frame_width": frame_width}, sort_keys=True)
    if marker.is_file() and marker.read_text(encoding="utf-8") == cache_key:
        return load_frame_samples(output_dir, sample_fps)
    frame_pattern = output_dir / "frame_%06d.jpg"
    for stale_frame in output_dir.glob("frame_*.jpg"):
        stale_frame.unlink()
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sample_fps},scale={frame_width}:-2",
            "-q:v",
            "3",
            str(frame_pattern),
        ]
    )

    samples = load_frame_samples(output_dir, sample_fps)
    marker.write_text(cache_key, encoding="utf-8")
    return samples


def load_frame_samples(output_dir: Path, sample_fps: float) -> list[FrameSample]:
    samples: list[FrameSample] = []
    previous: np.ndarray | None = None
    for index, path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        with Image.open(path) as image:
            current = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
        change = 0.0 if previous is None else float(np.mean(np.abs(current - previous)) / 255.0)
        samples.append(
            FrameSample(timestamp_sec=index / sample_fps, path=path, change_score=change)
        )
        previous = current
    if not samples:
        raise MediaError(f"Frame cache contains no images: {output_dir}")
    return samples


def _valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() > 0 and source.getframerate() == 16000
    except (wave.Error, EOFError):
        return False


def audio_energy_per_second(audio_path: Path) -> list[float]:
    if not audio_path.is_file():
        raise MediaError(f"Extracted audio is missing: {audio_path}")
    with wave.open(str(audio_path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        raw = source.readframes(source.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    energies: list[float] = []
    for start in range(0, len(samples), sample_rate):
        chunk = samples[start : start + sample_rate]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        energies.append(rms)
    return energies


def export_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{end_sec - start_sec:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def video_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _clean_title(stem: str) -> str:
    title = re.sub(r"^\([^)]*\)P\d+_第\d+集_#?", "", stem)
    title = title.split("【", 1)[0]
    title = re.sub(r"_(?:360|480|720|1080)P$", "", title, flags=re.IGNORECASE)
    return title.strip("_# ") or stem
