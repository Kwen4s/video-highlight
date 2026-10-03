from pathlib import Path

from scenedetect import AdaptiveDetector, detect

from ..models import SceneSegment


def detect_scenes(video_path: Path) -> list[SceneSegment]:
    pairs = detect(
        str(video_path),
        AdaptiveDetector(
            adaptive_threshold=3.0,
            min_scene_len=8,
            window_width=2,
            min_content_val=12.0,
        ),
        start_in_scene=True,
    )
    return [
        SceneSegment(
            start_sec=start.get_seconds(),
            end_sec=end.get_seconds(),
        )
        for start, end in pairs
        if end.get_seconds() > start.get_seconds()
    ]
