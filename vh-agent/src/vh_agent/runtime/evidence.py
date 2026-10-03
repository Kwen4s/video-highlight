"""Source-timed video observations for the ReAct agent.

The source timeline starts at the first displayed video frame. Clips are rendered
from the original media, retaining audio and relative audio/video timing. Requests
are expanded to video-frame boundaries; the returned source range is authoritative.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import VideoInfo
from ..preprocessing.media import MediaError, _run, probe_video, video_fingerprint

RENDER_PROFILE = {
    "version": 1,
    "video_codec": "libx264",
    "crf": 18,
    "preset": "veryfast",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "pixel_format": "yuv420p",
    "time_base": "1:1000000",
    "b_frames": 0,
    "preserve_tail_duration": True,
}


class VideoPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    page_id: str
    core_start_sec: float = Field(ge=0)
    core_end_sec: float = Field(gt=0)
    read_start_sec: float = Field(ge=0)
    read_end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> VideoPage:
        if not (
            self.read_start_sec <= self.core_start_sec < self.core_end_sec <= self.read_end_sec
        ):
            raise ValueError("Page read range must contain its nonempty core range")
        return self


class VideoObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    observation_id: str
    media_id: str
    requested_start_sec: float
    requested_end_sec: float
    src_start_sec: float
    src_end_sec: float
    path: Path
    has_audio: bool
    duration_sec: float
    page_id: str | None = None
    source_start_time_sec: float
    timestamp_precision_sec: float
    cache_hit: bool
    sampling_fps: float | None = Field(default=None, gt=0)

    def source_time(self, clip_time_sec: float) -> float:
        """Map a timestamp relative to the rendered clip to the source timeline."""
        if not math.isfinite(clip_time_sec) or not 0 <= clip_time_sec <= self.duration_sec:
            raise ValueError("Clip timestamp is outside the rendered observation")
        return min(self.src_end_sec, self.src_start_sec + clip_time_sec)


class EvidenceStore:
    def __init__(
        self,
        video: Path,
        output_dir: Path,
        page_sec: float = 30,
        overlap_sec: float = 3,
    ) -> None:
        if not math.isfinite(page_sec) or page_sec <= 0:
            raise ValueError("page_sec must be finite and positive")
        if not math.isfinite(overlap_sec) or overlap_sec < 0:
            raise ValueError("overlap_sec must be finite and nonnegative")
        info = probe_video(video)
        metadata = _probe(info.path)
        video_stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        self._frame_times, self.source_start_time_sec, duration = _video_timeline(
            info.path, video_stream, info.fps
        )
        self.video_info: VideoInfo = info.model_copy(update={"duration_sec": duration})
        self.media_id = f"media_{video_fingerprint(info.path)}"
        self.output_dir = output_dir.expanduser().resolve() / self.media_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pages = _pages(self.media_id, duration, page_sec, overlap_sec)
        self.overlap_sec = overlap_sec
        self._page_by_id = {page.page_id: page for page in self.pages}
        self._audio_streams = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
        self.metadata = {
            **self.video_info.model_dump(mode="json"),
            "media_id": self.media_id,
            "source_start_time_sec": self.source_start_time_sec,
            "timeline_origin": "first_displayed_video_frame",
            "render_profile": RENDER_PROFILE,
        }

    def split_page(self, page_id: str) -> list[VideoPage]:
        """Replace an oversized page with two complete, overlapping child pages."""
        try:
            parent = self._page_by_id[page_id]
        except KeyError as exc:
            raise ValueError(
                f"页面 {page_id} 不存在；请使用 progress.next_page 或 pages 中的 page_id。"
            ) from exc
        midpoint = (parent.core_start_sec + parent.core_end_sec) / 2
        if min(midpoint - parent.core_start_sec, parent.core_end_sec - midpoint) < 1:
            raise ValueError(
                "Video page cannot be split further: each child must cover at least 1 second"
            )
        children = []
        for start, end in (
            (parent.core_start_sec, midpoint),
            (midpoint, parent.core_end_sec),
        ):
            values = {
                "core_start_sec": start,
                "core_end_sec": end,
                "read_start_sec": max(parent.read_start_sec, start - self.overlap_sec),
                "read_end_sec": min(parent.read_end_sec, end + self.overlap_sec),
            }
            signature = json.dumps({"media_id": self.media_id, **values}, sort_keys=True)
            suffix = hashlib.sha256(signature.encode()).hexdigest()[:20]
            children.append(VideoPage(page_id=f"page_{suffix}", **values))
        index = self.pages.index(parent)
        self.pages[index : index + 1] = children
        self._page_by_id = {page.page_id: page for page in self.pages}
        return children

    def restore_pages(self, pages: list[VideoPage]) -> None:
        """Restore a checkpoint manifest without accepting gaps or lost coverage."""
        restored = [
            VideoPage.model_validate(page.model_dump() if isinstance(page, VideoPage) else page)
            for page in pages
        ]
        if not restored or len({page.page_id for page in restored}) != len(restored):
            raise ValueError("Page manifest must be nonempty and contain unique page IDs")
        expected_start = 0.0
        for page in restored:
            if not math.isclose(page.core_start_sec, expected_start, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Page core ranges must cover the video continuously in order")
            if page.read_end_sec > self.video_info.duration_sec + 1e-9:
                raise ValueError("Page read range exceeds the source video")
            expected_start = page.core_end_sec
        if not math.isclose(expected_start, self.video_info.duration_sec, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Page core ranges must cover the entire source video")
        self.pages = restored
        self._page_by_id = {page.page_id: page for page in restored}

    def scan(self, page_id: str) -> VideoObservation:
        try:
            page = self._page_by_id[page_id]
        except KeyError as exc:
            raise ValueError(
                f"页面 {page_id} 不存在；请使用 progress.next_page 或 pages 中的 page_id。"
            ) from exc
        observation = self.inspect(page.read_start_sec, page.read_end_sec)
        return observation.model_copy(
            update={
                "page_id": page_id,
                "observation_id": f"{observation.observation_id}:{page_id}",
            }
        )

    def inspect(self, start_sec: float, end_sec: float) -> VideoObservation:
        duration = self.video_info.duration_sec
        if not (
            math.isfinite(start_sec)
            and math.isfinite(end_sec)
            and 0 <= start_sec < end_sec <= duration
        ):
            raise ValueError(f"Expected 0 <= start_sec < end_sec <= {duration}")
        # Expanding to real packet PTS boundaries preserves the requested evidence.
        # The actual range is explicit, including when a request lies inside a frame.
        first = max(0, bisect.bisect_right(self._frame_times, start_sec) - 1)
        last = bisect.bisect_left(self._frame_times, end_sec)
        actual_start = self._frame_times[first]
        actual_end = self._frame_times[last] if last < len(self._frame_times) else duration
        signature = {
            "media_id": self.media_id,
            "start": actual_start,
            "end": actual_end,
            "render": RENDER_PROFILE,
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        path = self.output_dir / f"{digest}.mp4"
        cache_hit = path.is_file()
        if not cache_hit:
            self._render(path, actual_start, actual_end)
        rendered = _probe(path)
        rendered_duration, has_audio = self._validate_render(rendered, actual_start, actual_end)
        interval_times = [*self._frame_times[first:last], actual_end]
        gaps = [right - left for left, right in pairwise(interval_times)]
        precision = max(gaps, default=1 / self.video_info.fps)
        return VideoObservation(
            observation_id=f"obs_{digest}",
            media_id=self.media_id,
            requested_start_sec=start_sec,
            requested_end_sec=end_sec,
            src_start_sec=actual_start,
            src_end_sec=actual_end,
            path=path,
            has_audio=has_audio,
            duration_sec=rendered_duration,
            source_start_time_sec=self.source_start_time_sec,
            timestamp_precision_sec=precision,
            cache_hit=cache_hit,
        )

    def render_clip(self, start_sec: float, end_sec: float) -> VideoObservation:
        """Return the very same file that inspection and final playback should use."""
        return self.inspect(start_sec, end_sec)

    def frames(self, times: list[float], region=None, width: int = 1280) -> list[dict]:
        """Extract actual displayed frames once, retaining source PTS and optional ROI."""
        if any(not math.isfinite(t) or not 0 <= t < self.video_info.duration_sec for t in times):
            raise ValueError("帧时间必须在原片范围内。")
        indices = [max(0, bisect.bisect_right(self._frame_times, t) - 1) for t in times]
        paths = {}
        missing = []
        folder = self.output_dir / "frames"
        folder.mkdir(exist_ok=True)
        for index in sorted(set(indices)):
            key = hashlib.sha256(json.dumps([index, width, region]).encode()).hexdigest()[:24]
            path = folder / f"{key}.jpg"
            paths[index] = path
            if not path.is_file():
                missing.append(index)
        if missing:
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                selector = "+".join(f"eq(n,{index})" for index in missing)
                _run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(self.video_info.path),
                        "-vf",
                        f"select='{selector}',scale='min({width},iw)':-2",
                        "-fps_mode",
                        "vfr",
                        "-frames:v",
                        str(len(missing)),
                        "-q:v",
                        "2",
                        str(Path(temp) / "%05d.jpg"),
                    ]
                )
                decoded = sorted(Path(temp).glob("*.jpg"))
                if len(decoded) != len(missing):
                    raise ValueError("原片解码帧数与请求不一致。")
                for index, source in zip(missing, decoded, strict=True):
                    with Image.open(source) as image:
                        image.load()
                        if region:
                            w, h = image.size
                            image = image.crop(
                                (
                                    int(region[0] * w),
                                    int(region[1] * h),
                                    int(region[2] * w),
                                    int(region[3] * h),
                                )
                            )
                            image.save(source, quality=95)
                    source.replace(paths[index])
        return [
            {
                "requested_sec": t,
                "timestamp_sec": self._frame_times[index],
                "path": str(paths[index]),
                "region": region,
            }
            for t, index in zip(times, indices, strict=True)
        ]

    def _render(self, path: Path, start: float, end: float) -> None:
        absolute_start = self.source_start_time_sec + start
        absolute_end = self.source_start_time_sec + end
        last_frame = self._frame_times[bisect.bisect_left(self._frame_times, end) - 1]
        final_pts = last_frame - start
        final_duration = end - last_frame
        temporary = path.with_name(f".{path.stem}.{uuid4().hex}.mp4")
        # Seek by absolute PTS to the preceding keyframe, then perform the accurate
        # cut in the filters. FFmpeg's implicit accurate-seek trimming mixes the
        # format start offset with -copyts on some nonzero-start media.
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-copyts",
            "-noaccurate_seek",
            "-seek_timestamp",
            "1",
            "-ss",
            f"{absolute_start:.9f}",
            "-i",
            str(self.video_info.path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            (
                f"trim=start={absolute_start:.9f}:end={absolute_end:.9f},"
                f"setpts=PTS-{absolute_start:.9f}/TB,pad=ceil(iw/2)*2:ceil(ih/2)*2"
            ),
            "-c:v",
            RENDER_PROFILE["video_codec"],
            "-preset",
            RENDER_PROFILE["preset"],
            "-crf",
            str(RENDER_PROFILE["crf"]),
            "-pix_fmt",
            RENDER_PROFILE["pixel_format"],
            "-bf",
            str(RENDER_PROFILE["b_frames"]),
            "-fps_mode",
            "passthrough",
            "-enc_time_base:v",
            RENDER_PROFILE["time_base"],
            "-bsf:v",
            # Preserve the final frame's hold time; VFR decoders can report its
            # nominal codec duration instead. No B frames keeps PTS and DTS aligned.
            (
                "setts=pts=PTS:dts=DTS:duration="
                f"'if(gte(PTS,{final_pts:.9f}/TB-1),{final_duration:.9f}/TB,DURATION)'"
            ),
        ]
        if self._audio_streams:
            command.extend(
                [
                    "-af",
                    (
                        f"atrim=start={absolute_start:.9f}:end={absolute_end:.9f},"
                        f"asetpts=PTS-{absolute_start:.9f}/TB"
                    ),
                    "-c:a",
                    RENDER_PROFILE["audio_codec"],
                    "-b:a",
                    RENDER_PROFILE["audio_bitrate"],
                ]
            )
        command.extend(
            [
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        try:
            _run(command)
            self._validate_render(_probe(temporary), start, end)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_render(
        self, metadata: dict[str, Any], start: float, end: float
    ) -> tuple[float, bool]:
        streams = metadata.get("streams", [])
        video = next((s for s in streams if s["codec_type"] == "video"), None)
        if video is None:
            raise MediaError("Rendered observation contains no video stream")
        rendered_start = _number(video.get("start_time"), 0)
        rendered_duration = _number(metadata.get("format", {}).get("duration"), 0)
        video_duration = _number(video.get("duration"), 0)
        tolerance = 0.003
        if (
            abs(rendered_start) > 0.003
            or rendered_duration <= 0
            or abs(video_duration - (end - start)) > tolerance
            or abs(rendered_duration - (end - start)) > tolerance
        ):
            raise MediaError("Rendered observation does not preserve the requested source timeline")
        audio = [s for s in streams if s["codec_type"] == "audio"]
        expected_audio = [
            s
            for s in self._audio_streams
            if _stream_overlaps(
                s, self.source_start_time_sec + start, self.source_start_time_sec + end
            )
        ]
        if len(audio) < len(expected_audio):
            raise MediaError("Rendering lost an audio stream present in the source interval")
        return rendered_duration, bool(audio)


def _probe(path: Path) -> dict[str, Any]:
    return json.loads(
        _run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        ).stdout
    )


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _video_timeline(
    path: Path, stream: dict[str, Any], fps: float
) -> tuple[list[float], float, float]:
    payload = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,duration_time,flags",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    declared_start = _number(stream.get("start_time"), -math.inf)
    packets: dict[float, float] = {}
    for packet in payload.get("packets", []):
        pts = _number(packet.get("pts_time"), math.nan)
        if math.isfinite(pts) and pts >= declared_start and "D" not in packet.get("flags", ""):
            packets[pts] = _number(packet.get("duration_time"), 0)
    if not packets:
        raise MediaError("Video has no usable presentation timestamps")
    times = sorted(packets)
    origin = times[0]
    final_duration = packets[times[-1]]
    if final_duration <= 0:
        declared_end = declared_start + _number(stream.get("duration"), 0)
        final_duration = declared_end - times[-1] if declared_end > times[-1] else 1 / fps
    duration = round(times[-1] + final_duration - origin, 9)
    return [round(t - origin, 9) for t in times], origin, duration


def _stream_overlaps(stream: dict[str, Any], start: float, end: float) -> bool:
    stream_start = _number(stream.get("start_time"), -math.inf)
    duration = _number(stream.get("duration"), math.inf)
    stream_end = stream_start + duration if math.isfinite(stream_start) else math.inf
    return stream_start < end and stream_end > start


def _pages(media_id: str, duration: float, page_sec: float, overlap_sec: float) -> list[VideoPage]:
    pages: list[VideoPage] = []
    for index in range(math.ceil(duration / page_sec)):
        start = index * page_sec
        end = min(duration, (index + 1) * page_sec)
        signature = f"{media_id}:{start:.9f}:{end:.9f}:{overlap_sec:.9f}"
        suffix = hashlib.sha256(signature.encode()).hexdigest()[:12]
        pages.append(
            VideoPage(
                page_id=f"page_{index:05d}_{suffix}",
                core_start_sec=start,
                core_end_sec=end,
                read_start_sec=max(0, start - overlap_sec),
                read_end_sec=min(duration, end + overlap_sec),
            )
        )
    return pages
