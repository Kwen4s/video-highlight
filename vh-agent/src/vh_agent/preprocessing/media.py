import hashlib
import json
import subprocess
import wave
from pathlib import Path

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


def probe_video(path: Path) -> VideoInfo:
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
    samples = [
        FrameSample(timestamp_sec=index / sample_fps, path=path)
        for index, path in enumerate(sorted(output_dir.glob("frame_*.jpg")))
    ]
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


def video_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
