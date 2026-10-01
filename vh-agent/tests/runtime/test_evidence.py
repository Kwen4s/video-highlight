import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vh_agent.preprocessing.media import MediaError
from vh_agent.runtime.evidence import EvidenceStore, VideoPage

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Video evidence integration tests require FFmpeg and FFprobe",
)


def _make_video(path: Path, *, audio: bool = True, offset: float = 0, audio_delay: float = 0):
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            "color=red:s=96x64:r=10:d=6,"
            "drawbox=color=blue:t=fill:enable='gte(t,2)',"
            "drawbox=color=green:t=fill:enable='gte(t,4)'"
        ),
    ]
    if audio:
        command.extend(
            [
                "-itsoffset",
                str(audio_delay),
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=500:sample_rate=48000:duration={6 - audio_delay}",
                "-c:a",
                "aac",
            ]
        )
    command.extend(
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-output_ts_offset", str(offset), str(path)]
    )
    subprocess.run(command, check=True, capture_output=True)
    return path


def _probe(path: Path):
    return json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        )
    )


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _make_video(tmp_path_factory.mktemp("video_evidence") / "source.mp4", offset=5)


def test_page_manifest_covers_tail_and_scan_keeps_overlap(source, tmp_path):
    store = EvidenceStore(source, tmp_path, page_sec=2.5, overlap_sec=0.3)
    assert store.video_info.duration_sec == pytest.approx(6)
    assert [(p.core_start_sec, p.core_end_sec) for p in store.pages] == [(0, 2.5), (2.5, 5), (5, 6)]
    assert [(p.read_start_sec, p.read_end_sec) for p in store.pages] == [
        (0, 2.8),
        (2.2, 5.3),
        (4.7, 6),
    ]
    observation = store.scan(store.pages[-1].page_id)
    assert observation.page_id == store.pages[-1].page_id
    assert observation.src_start_sec <= 5
    assert observation.src_end_sec == pytest.approx(6)
    assert observation.duration_sec == pytest.approx(1.3, abs=0.002)
    with pytest.raises(ValueError, match="页面.*不存在"):
        store.scan("unregistered_page")


def test_nonzero_pts_clip_reads_actual_source_and_preserves_audio(source, tmp_path):
    store = EvidenceStore(source, tmp_path)
    observation = store.inspect(2.03, 3.57)
    assert store.source_start_time_sec == pytest.approx(5)
    assert observation.requested_start_sec == 2.03
    assert observation.requested_end_sec == 3.57
    assert observation.src_start_sec == pytest.approx(2)
    assert observation.src_end_sec == pytest.approx(3.6)
    assert observation.duration_sec == pytest.approx(1.6, abs=0.002)
    assert observation.source_time(0.5) == pytest.approx(2.5)
    assert observation.has_audio
    pixels = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(observation.path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    color = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 3).mean(axis=0)
    assert color[2] > 240 and color[0] < 15 and color[1] < 15  # blue at source t=2
    sound = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(observation.path),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "pipe:1",
        ]
    )
    assert np.abs(np.frombuffer(sound, dtype=np.int16).astype(float)).mean() > 100


def test_rendered_clip_is_reused_for_final_review_and_publication(source, tmp_path):
    store = EvidenceStore(source, tmp_path)
    observed = store.inspect(2, 3.6)
    modified = observed.path.stat().st_mtime_ns
    reopened = EvidenceStore(source, tmp_path).render_clip(2, 3.6)
    assert observed.observation_id == reopened.observation_id
    assert observed.path == reopened.path
    assert reopened.cache_hit
    assert reopened.path.stat().st_mtime_ns == modified
    assert EvidenceStore(source, tmp_path).pages == store.pages


def test_page_observations_are_distinct_when_they_share_the_same_media(source, tmp_path):
    store = EvidenceStore(source, tmp_path, page_sec=3, overlap_sec=6)
    first = store.scan(store.pages[0].page_id)
    second = store.scan(store.pages[1].page_id)
    inspected = store.inspect(0, 6)
    assert first.path == second.path == inspected.path
    assert len({first.observation_id, second.observation_id, inspected.observation_id}) == 3
    assert first.page_id == store.pages[0].page_id
    assert second.page_id == store.pages[1].page_id
    assert inspected.page_id is None


def test_page_split_preserves_coverage_and_clamps_overlap_to_parent(source, tmp_path):
    store = EvidenceStore(source, tmp_path, page_sec=4, overlap_sec=0.5)
    parent, tail = store.pages
    children = store.split_page(parent.page_id)
    assert [(p.core_start_sec, p.core_end_sec) for p in store.pages] == [(0, 2), (2, 4), (4, 6)]
    assert [(p.read_start_sec, p.read_end_sec) for p in children] == [(0, 2.5), (1.5, 4.5)]
    assert store.pages[-1] == tail
    assert store.scan(children[1].page_id).page_id == children[1].page_id
    with pytest.raises(ValueError, match="页面.*不存在"):
        store.scan(parent.page_id)
    recreated = EvidenceStore(source, tmp_path, page_sec=4, overlap_sec=0.5)
    assert recreated.split_page(recreated.pages[0].page_id) == children
    grandchildren = store.split_page(children[1].page_id)
    assert [(p.read_start_sec, p.read_end_sec) for p in grandchildren] == [(1.5, 3.5), (2.5, 4.5)]
    before = list(store.pages)
    with pytest.raises(ValueError, match="at least 1 second"):
        store.split_page(grandchildren[0].page_id)
    assert store.pages == before


def test_split_manifest_can_be_restored_but_not_gapped_or_truncated(source, tmp_path):
    store = EvidenceStore(source, tmp_path, page_sec=4, overlap_sec=0.5)
    store.split_page(store.pages[0].page_id)
    saved = [VideoPage.model_validate_json(page.model_dump_json()) for page in store.pages]
    restored = EvidenceStore(source, tmp_path, page_sec=4, overlap_sec=0.5)
    restored.restore_pages(saved)
    assert restored.pages == store.pages
    assert restored.scan(saved[0].page_id).page_id == saved[0].page_id
    invalid_manifests = [
        [],
        saved[:-1],
        [saved[0], saved[0], *saved[1:]],
        [saved[0], saved[2], saved[1]],
        [
            saved[0],
            VideoPage(
                **{
                    **saved[1].model_dump(),
                    "core_start_sec": 2.1,
                }
            ),
            saved[2],
        ],
        [
            *saved[:-1],
            VideoPage(
                **{
                    **saved[-1].model_dump(),
                    "read_end_sec": 7,
                }
            ),
        ],
    ]
    for pages in invalid_manifests:
        with pytest.raises(ValueError):
            restored.restore_pages(pages)
        assert restored.pages == saved


def test_silent_video_is_valid_and_audio_delay_is_not_reset(tmp_path):
    silent = EvidenceStore(_make_video(tmp_path / "silent.mp4", audio=False), tmp_path / "silent")
    observed = silent.inspect(1, 3)
    assert not observed.has_audio
    assert [s["codec_type"] for s in _probe(observed.path)["streams"]] == ["video"]

    delayed_path = _make_video(tmp_path / "delayed.mp4", offset=5, audio_delay=2)
    delayed = EvidenceStore(delayed_path, tmp_path / "delayed")
    observed = delayed.inspect(1, 4)
    source_audio = next(s for s in _probe(delayed_path)["streams"] if s["codec_type"] == "audio")
    clip_audio = next(s for s in _probe(observed.path)["streams"] if s["codec_type"] == "audio")
    expected_start = float(source_audio["start_time"]) - delayed.source_start_time_sec - 1
    # AAC priming can add an encoder packet; relative timing remains within one packet.
    assert float(clip_audio["start_time"]) == pytest.approx(expected_start, abs=0.025)


def test_missing_audio_in_cached_observation_is_not_silently_accepted(source, tmp_path):
    store = EvidenceStore(source, tmp_path)
    observation = store.inspect(2, 3)
    silent = tmp_path / "corrupted.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(observation.path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            str(silent),
        ],
        check=True,
        capture_output=True,
    )
    silent.replace(observation.path)
    with pytest.raises(MediaError, match="lost an audio stream"):
        store.inspect(2, 3)


def test_variable_frame_spacing_uses_real_timestamps(source, tmp_path):
    variable = tmp_path / "variable.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "select='not(mod(n,3))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-an",
            str(variable),
        ],
        check=True,
        capture_output=True,
    )
    store = EvidenceStore(variable, tmp_path / "evidence")
    observation = store.inspect(2.03, 3.57)
    assert observation.src_start_sec == pytest.approx(1.8)
    assert observation.src_end_sec == pytest.approx(3.6)
    assert observation.timestamp_precision_sec == pytest.approx(0.3)
    assert observation.duration_sec == pytest.approx(1.8, abs=0.002)


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (0, 7), (2, 2), (3, 2), (float("nan"), 2), (0, float("inf"))]
)
def test_invalid_intervals_are_rejected_without_rendering(source, tmp_path, start, end):
    store = EvidenceStore(source, tmp_path)
    with pytest.raises(ValueError, match="Expected"):
        store.inspect(start, end)
    assert not list(store.output_dir.glob("*.mp4"))


@pytest.mark.parametrize("page,overlap", [(0, 3), (float("nan"), 3), (30, -1), (30, float("inf"))])
def test_invalid_page_configuration_is_rejected(source, tmp_path, page, overlap):
    with pytest.raises(ValueError):
        EvidenceStore(source, tmp_path, page_sec=page, overlap_sec=overlap)
