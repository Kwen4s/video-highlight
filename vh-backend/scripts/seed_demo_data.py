from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings  # noqa: E402
from app.repository import JobRepository  # noqa: E402


@dataclass(frozen=True)
class DemoHighlight:
    highlight_id: str
    start_sec: float
    end_sec: float
    score: float
    highlight_type: str
    description: str
    reason: str
    scene_label: str


@dataclass(frozen=True)
class DemoJob:
    job_id: str
    original_name: str
    title: str
    subtitle: str
    backgrounds: tuple[str, str, str]
    accent: str
    accent_secondary: str
    audio_frequency: int
    highlights: tuple[DemoHighlight, DemoHighlight, DemoHighlight]


DEMO_JOBS = (
    DemoJob(
        job_id="job_demo_citypulse",
        original_name="城市节拍_原片.mp4",
        title="CITY PULSE",
        subtitle="A THREE ACT URBAN STUDY",
        backgrounds=("0x102433", "0x3b1e20", "0x14291e"),
        accent="0xd6ff3f",
        accent_secondary="0x4fd1ff",
        audio_frequency=220,
        highlights=(
            DemoHighlight(
                highlight_id="hl_city_arrival",
                start_sec=1,
                end_sec=5,
                score=0.96,
                highlight_type="opening",
                description="清晨抵达",
                reason="冷色画面与快速运动构成清晰的开场节奏。",
                scene_label="01 / MORNING ARRIVAL",
            ),
            DemoHighlight(
                highlight_id="hl_city_crossing",
                start_sec=7,
                end_sec=11,
                score=0.92,
                highlight_type="motion",
                description="街口穿行",
                reason="暖色切换和交错运动让画面形成明显动势。",
                scene_label="02 / RUSH HOUR CROSSING",
            ),
            DemoHighlight(
                highlight_id="hl_city_night",
                start_sec=13,
                end_sec=17,
                score=0.89,
                highlight_type="atmosphere",
                description="夜色落点",
                reason="节奏放缓，绿色光块为原片提供稳定收束。",
                scene_label="03 / NIGHT SIGNAL",
            ),
        ),
    ),
    DemoJob(
        job_id="job_demo_launchfilm",
        original_name="新品发布_原片.mp4",
        title="FORM / FUNCTION",
        subtitle="PRODUCT LAUNCH CANVAS",
        backgrounds=("0x281b36", "0x132b35", "0x392b12"),
        accent="0xff6f61",
        accent_secondary="0xffd166",
        audio_frequency=330,
        highlights=(
            DemoHighlight(
                highlight_id="hl_launch_reveal",
                start_sec=1,
                end_sec=5,
                score=0.95,
                highlight_type="reveal",
                description="轮廓初现",
                reason="中心构图与高对比色块完成第一次产品揭示。",
                scene_label="01 / SILHOUETTE REVEAL",
            ),
            DemoHighlight(
                highlight_id="hl_launch_detail",
                start_sec=7,
                end_sec=11,
                score=0.91,
                highlight_type="detail",
                description="细节特写",
                reason="画面密度提升，适合作为功能细节的展示段落。",
                scene_label="02 / MATERIAL DETAIL",
            ),
            DemoHighlight(
                highlight_id="hl_launch_finale",
                start_sec=13,
                end_sec=17,
                score=0.87,
                highlight_type="finale",
                description="发布定格",
                reason="金色背景和完整构图形成有记忆点的结束画面。",
                scene_label="03 / HERO FINALE",
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create two local demo video-highlight jobs.")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path("ffmpeg"),
        help="Path to ffmpeg.exe (defaults to ffmpeg on PATH).",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("C:/Windows/Fonts/segoeuib.ttf"),
        help="Font used for the canvas labels.",
    )
    return parser.parse_args()


def filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:")


def canvas_filter(job: DemoJob, font: Path) -> str:
    font_option = f"fontfile='{filter_path(font)}'" if font.is_file() else "font='Arial'"
    filters = [
        (
            f"drawbox=x=0:y=0:w=iw:h=ih:color={background}:t=fill:"
            f"enable='between(t,{index * 6},{(index + 1) * 6})'"
        )
        for index, background in enumerate(job.backgrounds)
    ]
    filters.extend(
        [
            "drawgrid=w=80:h=80:t=1:color=white@0.08",
            (
                "drawbox=x='70+58*t':y='170+55*sin(t*1.8)':w=190:h=190:"
                f"color={job.accent}@0.92:t=fill"
            ),
            (
                "drawbox=x='1100-46*t':y='420+45*cos(t*1.5)':w=130:h=130:"
                f"color={job.accent_secondary}@0.82:t=fill"
            ),
            "drawbox=x=64:y=55:w=5:h=94:color=white@0.9:t=fill",
            "drawbox=x=64:y=650:w=1152:h=3:color=white@0.2:t=fill",
            (
                f"drawbox=x=64:y=650:w='64*t':h=3:color={job.accent}:t=fill"
            ),
            (
                f"drawtext={font_option}:text='{job.title}':fontcolor=white:"
                "fontsize=45:x=88:y=60"
            ),
            (
                f"drawtext={font_option}:text='{job.subtitle}':fontcolor=white@0.58:"
                "fontsize=18:x=91:y=118"
            ),
        ]
    )
    for index, highlight in enumerate(job.highlights):
        filters.append(
            f"drawtext={font_option}:text='{highlight.scene_label}':fontcolor=white:"
            f"fontsize=25:x=72:y=595:enable='between(t,{index * 6},{(index + 1) * 6})'"
        )
    return ",".join(filters)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def create_source(ffmpeg: Path, font: Path, job: DemoJob, target: Path) -> None:
    run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0b0e0d:s=1280x720:r=30:d=18",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={job.audio_frequency}:sample_rate=48000:duration=18",
            "-vf",
            canvas_filter(job, font),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(target),
        ]
    )


def create_clip(ffmpeg: Path, source: Path, highlight: DemoHighlight, target: Path) -> None:
    run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            str(highlight.start_sec),
            "-t",
            str(highlight.end_sec - highlight.start_sec),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def make_result(job: DemoJob) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "job_id": job.job_id,
        "video": {"video_id": job.job_id, "title": job.original_name, "duration_sec": 18},
        "highlights": [
            {
                "highlight_id": highlight.highlight_id,
                "start_sec": highlight.start_sec,
                "end_sec": highlight.end_sec,
                "score": highlight.score,
                "highlight_type": highlight.highlight_type,
                "description": highlight.description,
                "reason": highlight.reason,
                "clip_url": f"clips/{highlight.highlight_id}.mp4",
                "review_status": "pending",
            }
            for highlight in job.highlights
        ],
    }


def seed_job(
    *,
    ffmpeg: Path,
    font: Path,
    settings: Settings,
    repository: JobRepository,
    job: DemoJob,
) -> str:
    existing = repository.get(job.job_id)
    target_dir = settings.jobs_dir / job.job_id
    if existing:
        return f"Skipped {job.original_name}: demo job already exists"
    if target_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing directory: {target_dir}")

    staging_root = settings.storage_dir / ".seed-staging"
    staging_dir = staging_root / job.job_id
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    source_dir = staging_dir / "source"
    clips_dir = staging_dir / "clips"
    source_dir.mkdir(parents=True)
    clips_dir.mkdir()

    try:
        source_path = source_dir / "original.mp4"
        create_source(ffmpeg, font, job, source_path)
        for highlight in job.highlights:
            create_clip(
                ffmpeg,
                source_path,
                highlight,
                clips_dir / f"{highlight.highlight_id}.mp4",
            )

        result = make_result(job)
        (staging_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        staging_dir.replace(target_dir)
        repository.create(
            job_id=job.job_id,
            original_name=job.original_name,
            stored_name="original.mp4",
            content_type="video/mp4",
            size_bytes=(target_dir / "source" / "original.mp4").stat().st_size,
            language="zh",
        )
        repository.save_result(job.job_id, result)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    finally:
        if staging_root.exists() and not any(staging_root.iterdir()):
            staging_root.rmdir()

    return f"Created {job.original_name}: 1 source + {len(job.highlights)} highlights"


def main() -> int:
    args = parse_args()
    ffmpeg = (
        args.ffmpeg.resolve()
        if args.ffmpeg.exists()
        else Path(shutil.which(str(args.ffmpeg)) or "")
    )
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg was not found: {args.ffmpeg}")

    settings = Settings()
    repository = JobRepository(settings.database_path)
    repository.initialize()
    for job in DEMO_JOBS:
        print(
            seed_job(
                ffmpeg=ffmpeg,
                font=args.font,
                settings=settings,
                repository=repository,
                job=job,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
