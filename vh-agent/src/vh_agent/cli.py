import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .evaluation import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    run_evaluation,
    score_evaluation,
)
from .highlight_model import HighlightModelConfig, fit_highlight_model
from .models import DetectionResult, DetectionTask
from .pipeline import HighlightDetectionService
from .preprocessing.media import probe_video
from .silver_labeling import (
    DEFAULT_METADATA_DIR,
    DEFAULT_SILVER_DATASET_DIR,
    run_silver_labeling,
)

app = typer.Typer(no_args_is_help=True, help="Short-drama highlight detection pipeline")
evaluation_app = typer.Typer(no_args_is_help=True, help="Run and score the test dataset")
label_app = typer.Typer(no_args_is_help=True, help="Produce resumable silver labels")
train_app = typer.Typer(no_args_is_help=True, help="Train the local multimodal highlighter")
app.add_typer(evaluation_app, name="evaluate")
app.add_typer(label_app, name="label")
app.add_typer(train_app, name="train")
console = Console()


@app.command()
def inspect(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    language: str | None = typer.Option(None, "--language", "-l"),
) -> None:
    """Inspect a video's media metadata without running inference."""
    info = probe_video(video, language)
    console.print_json(json.dumps(info.model_dump(mode="json"), ensure_ascii=False))


@app.command()
def run(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    language: str | None = typer.Option(None, "--language", "-l"),
    video_id: str | None = typer.Option(None, "--video-id"),
    job_id: str | None = typer.Option(None, "--job-id"),
    trace: bool = typer.Option(False, "--trace", help="Write one internal trace.json file."),
) -> None:
    """Run the complete pipeline for one video."""
    settings = Settings(VH_WRITE_TRACE=trace)
    task_values: dict[str, object] = {
        "video_path": video,
        "video_id": video_id,
        "language": language,
    }
    if job_id is not None:
        task_values["job_id"] = job_id
    result = HighlightDetectionService(settings).detect(DetectionTask(**task_values))

    table = Table(title=f"Highlights: {result.video.title}")
    table.add_column("Time")
    table.add_column("Score", justify="right")
    table.add_column("Type")
    table.add_column("Description")
    for item in result.highlights:
        table.add_row(
            f"{item.start_sec:.1f}-{item.end_sec:.1f}s",
            f"{item.score:.3f}",
            item.highlight_type,
            item.description,
        )
    console.print(table)
    console.print(f"job_id={result.job_id}")
    console.print(f"result={settings.job_output_dir / result.job_id / 'result.json'}")


@evaluation_app.command("run")
def evaluate_run(
    run_id: str = typer.Option("test_v1", "--run-id"),
    resume: bool = typer.Option(False, "--resume"),
    limit: int | None = typer.Option(None, min=1),
    dataset_dir: Path = typer.Option(
        DEFAULT_DATASET_DIR,
        exists=True,
        file_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, file_okay=False),
) -> None:
    """Run highlight detection over the test dataset."""
    predictions = run_evaluation(
        run_id,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        resume=resume,
        limit=limit,
    )
    console.print(f"predictions={predictions}")


@evaluation_app.command("score")
def evaluate_score(
    run_id: str = typer.Option("test_v1", "--run-id"),
    include_silver: bool = typer.Option(False, "--include-silver"),
    dataset_dir: Path = typer.Option(
        DEFAULT_DATASET_DIR,
        exists=True,
        file_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, file_okay=False),
) -> None:
    """Score saved predictions against test annotations."""
    metrics = score_evaluation(
        run_id,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        include_silver=include_silver,
    )
    console.print_json(json.dumps(metrics, ensure_ascii=False))


@label_app.command("run")
def label_run(
    run_id: str = typer.Option("gemini37_scene_v1", "--run-id"),
    resume: bool = typer.Option(True, "--resume/--restart"),
    limit: int | None = typer.Option(None, min=1),
    metadata_dir: Path = typer.Option(
        DEFAULT_METADATA_DIR,
        exists=True,
        file_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(DEFAULT_SILVER_DATASET_DIR, file_okay=False),
) -> None:
    """Label all source metadata records with Gemini 3.7, without exporting clips."""
    annotations = run_silver_labeling(
        run_id,
        metadata_dir=metadata_dir,
        output_dir=output_dir,
        resume=resume,
        limit=limit,
    )
    console.print(f"annotations={annotations}")


@train_app.command("run")
def train_run(
    silver_run_id: str = typer.Option("gemini37_transition_v1", "--silver-run-id"),
    annotations: Path | None = typer.Option(None, "--annotations"),
    output_dir: Path = typer.Option(
        Path("outputs/highlight_model/seed_7_supv2"), "--output-dir"
    ),
    vision_model_path: Path = typer.Option(
        Path("/data1/modelscope_models/Qwen3-VL-Embedding-2B"),
        "--vision-model-path",
        exists=True,
    ),
    audio_model_path: Path = typer.Option(
        Path(
            "/data1/video-highlight-models/modelscope/models/"
            "iic--SenseVoiceSmall/snapshots/master"
        ),
        "--audio-model-path",
        exists=True,
    ),
    media_cache_dir: Path = typer.Option(
        Path("/data1/video-highlight-cache"), "--media-cache-dir", exists=True
    ),
    feature_cache_dir: Path = typer.Option(
        Path("/data1/video-highlight-model-features"), "--feature-cache-dir"
    ),
    stage: str = typer.Option("all", "--stage", help="features, train, or all"),
    device: str = typer.Option("cuda:0", "--device"),
    feature_device: str = typer.Option("cuda:0", "--feature-device"),
    epochs: int = typer.Option(20, "--epochs", min=1),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", min=1e-7),
    gradient_accumulation: int = typer.Option(4, "--gradient-accumulation", min=1),
    model_dim: int = typer.Option(512, "--model-dim", min=128),
    attention_heads: int = typer.Option(8, "--attention-heads", min=1),
    temporal_layers_per_level: int = typer.Option(
        2, "--temporal-layers-per-level", min=1
    ),
    vision_batch_size: int = typer.Option(8, "--vision-batch-size", min=1),
    seed: int = typer.Option(7, "--seed"),
    init_checkpoint: Path | None = typer.Option(None, "--init-checkpoint"),
    finetune_heads: bool = typer.Option(False, "--finetune-heads"),
) -> None:
    """Cache frozen multimodal moments and train the narrative-transition localizer."""
    if stage not in {"features", "train", "all"}:
        raise typer.BadParameter("stage must be features, train, or all")
    source = annotations or DEFAULT_SILVER_DATASET_DIR / silver_run_id / "annotations.jsonl"
    if not source.is_file():
        raise typer.BadParameter(f"silver annotations not found: {source}")
    report = fit_highlight_model(
        HighlightModelConfig(
            annotations=source,
            output_dir=output_dir,
            vision_model_path=vision_model_path,
            audio_model_path=audio_model_path,
            media_cache_dir=media_cache_dir,
            feature_cache_dir=feature_cache_dir,
            stage=stage,
            device=device,
            feature_device=feature_device,
            epochs=epochs,
            learning_rate=learning_rate,
            gradient_accumulation=gradient_accumulation,
            model_dim=model_dim,
            attention_heads=attention_heads,
            temporal_layers_per_level=temporal_layers_per_level,
            vision_batch_size=vision_batch_size,
            seed=seed,
            init_checkpoint=init_checkpoint,
            finetune_heads=finetune_heads,
        )
    )
    console.print_json(json.dumps(report, ensure_ascii=False))


@app.command()
def review(
    result_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    highlight_id: str = typer.Argument(...),
    status: str = typer.Option(..., help="accepted, rejected, or revised"),
    start_sec: float | None = typer.Option(None),
    end_sec: float | None = typer.Option(None),
    labels_path: Path = typer.Option(Path("outputs/reviews.jsonl")),
) -> None:
    """Record human feedback as a training label."""
    if status not in {"accepted", "rejected", "revised"}:
        raise typer.BadParameter("status must be accepted, rejected, or revised")

    result = DetectionResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    target = next(
        (item for item in result.highlights if item.highlight_id == highlight_id),
        None,
    )
    if target is None:
        raise typer.BadParameter(f"highlight not found: {highlight_id}")

    reviewed_start = target.start_sec if start_sec is None else max(0.0, start_sec)
    reviewed_end = target.end_sec if end_sec is None else min(result.video.duration_sec, end_sec)
    if reviewed_end <= reviewed_start:
        raise typer.BadParameter("end_sec must be greater than start_sec")

    target.review_status = status
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    row = target.model_dump(mode="json")
    row.update(
        {
            "job_id": result.job_id,
            "video_id": result.video.video_id,
            "video_title": result.video.title,
            "start_sec": reviewed_start,
            "end_sec": reviewed_end,
        }
    )
    with labels_path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"Recorded [bold]{status}[/bold] for {highlight_id} -> {labels_path}")


if __name__ == "__main__":
    app()
