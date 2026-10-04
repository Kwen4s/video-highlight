import json
from pathlib import Path

import typer
from rich.console import Console

from .config import Settings
from .evaluation import (
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_DIR,
    run_evaluation,
    score_evaluation,
)
from .models import DetectionResult, DetectionTask
from .pipeline import HighlightDetectionService
from .preprocessing.media import probe_video

app = typer.Typer(
    no_args_is_help=True,
    help="Short-drama highlight detection pipeline",
    pretty_exceptions_show_locals=False,
)
evaluation_app = typer.Typer(no_args_is_help=True, help="Run and score the test dataset")
train_app = typer.Typer(no_args_is_help=True, help="Train the local multimodal highlighter")
app.add_typer(evaluation_app, name="evaluate")
app.add_typer(train_app, name="train")
console = Console()


@app.command()
def inspect(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Inspect a video's media metadata without running inference."""
    info = probe_video(video)
    console.print_json(json.dumps(info.model_dump(mode="json"), ensure_ascii=False))


@app.command()
def run(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    language: str | None = typer.Option(None, "--language", "-l"),
    video_id: str | None = typer.Option(None, "--video-id"),
    job_id: str | None = typer.Option(None, "--job-id"),
    task_file: Path | None = typer.Option(None, "--task-file", exists=True),
    instruction: str | None = typer.Option(None, "--instruction"),
    subtitles: Path | None = typer.Option(None, "--subtitles", exists=True),
    max_clip_sec: float | None = typer.Option(None, "--max-clip-sec", min=0.1),
    max_highlights: int | None = typer.Option(None, "--max-highlights", min=1),
    resume: bool = typer.Option(False, "--resume"),
    verbose: bool = typer.Option(False, "--verbose", help="显示逐步工具进度"),
) -> None:
    """Analyze a video; interrupted execution exits 1 with its checkpoint saved."""
    values = json.loads(task_file.read_text()) if task_file else {}
    values.update(video_path=video, resume=resume)
    for key, value in {
        "language": language,
        "video_id": video_id,
        "job_id": job_id,
        "instruction": instruction,
        "subtitle_path": subtitles,
        "max_clip_sec": max_clip_sec,
        "max_highlights": max_highlights,
    }.items():
        if value is not None:
            values[key] = value
    settings = Settings()
    result = HighlightDetectionService(
        settings,
        on_progress=console.print if verbose else None,
    ).detect(DetectionTask(**values))
    console.print(f"completion={result.completion}, highlights={len(result.highlights)}")
    console.print(f"result={settings.job_output_dir / result.job_id / 'result.json'}")
    if result.completion != "complete":
        console.print(f"分析尚未完成（{result.analysis.stop_reason}），可从检查点继续。")
        if result.analysis.stop_reason == "execution_error":
            raise typer.Exit(code=1)
        raise typer.Exit(code=2)


@evaluation_app.command("run")
def evaluate_run(
    run_id: str = typer.Option(..., "--run-id"),
    resume: bool = typer.Option(False, "--resume"),
    limit: int | None = typer.Option(None, min=1),
    include_silver: bool = typer.Option(False, "--include-silver"),
    video_ids: list[str] | None = typer.Option(None, "--video-id"),
    task_file: Path | None = typer.Option(None, "--task-file", exists=True),
    dataset_dir: Path = typer.Option(DEFAULT_DATASET_DIR, exists=True, file_okay=False),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, file_okay=False),
) -> None:
    """Freeze and evaluate labeled videos; resume reuses the saved protocol."""
    report = run_evaluation(
        run_id,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        resume=resume,
        limit=limit,
        include_silver=include_silver,
        video_ids=video_ids,
        task_file=task_file,
    )
    console.print(f"report={report}")


@evaluation_app.command("score")
def evaluate_score(
    run_id: str = typer.Option(..., "--run-id"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, file_okay=False),
) -> None:
    """Rebuild metrics offline using frozen annotations and saved job results."""
    metrics = score_evaluation(run_id, output_dir=output_dir)
    console.print_json(
        json.dumps({k: v for k, v in metrics.items() if k != "details"}, ensure_ascii=False)
    )


@train_app.command("run")
def train_run(
    annotations: Path = typer.Option(
        ..., "--annotations", exists=True, dir_okay=False, readable=True
    ),
    output_dir: Path = typer.Option(Path("outputs/highlight_model"), "--output-dir"),
    vision_model_path: Path = typer.Option(
        Path("/data1/modelscope_models/Qwen3-VL-Embedding-2B"),
        "--vision-model-path",
        exists=True,
    ),
    audio_model_path: Path = typer.Option(
        Path(
            "/data1/video-highlight-models/modelscope/models/iic--SenseVoiceSmall/snapshots/master"
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
    temporal_layers_per_level: int = typer.Option(2, "--temporal-layers-per-level", min=1),
    vision_batch_size: int = typer.Option(8, "--vision-batch-size", min=1),
    seed: int = typer.Option(7, "--seed"),
    init_checkpoint: Path | None = typer.Option(None, "--init-checkpoint"),
    finetune_heads: bool = typer.Option(False, "--finetune-heads"),
) -> None:
    """Cache frozen multimodal moments and train the narrative-transition localizer."""
    from .highlight_model import HighlightModelConfig, fit_highlight_model

    if stage not in {"features", "train", "all"}:
        raise typer.BadParameter("stage must be features, train, or all")
    report = fit_highlight_model(
        HighlightModelConfig(
            annotations=annotations,
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
    """Record an editor's decision on a recommended clip."""
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
    if (target.start_sec, target.end_sec) != (reviewed_start, reviewed_end):
        target.start_sec, target.end_sec = reviewed_start, reviewed_end
        target.clip_url = ""
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
