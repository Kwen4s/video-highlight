"""Commands for source intake, labeling, review, export and detector feedback."""

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import Settings
from .exporter import export
from .labeler import run
from .models import VideoPredictions
from .store import Store

app = typer.Typer(help="短剧检测模型的数据飞轮", pretty_exceptions_enable=False)


def output(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command()
def ingest(manifests: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)]):
    """导入视频清单，按内容去重、按短剧划分训练/验证/测试。"""
    output(Store(Settings().root).ingest(manifests))


@app.command("run")
def produce(
    limit: Annotated[int, typer.Option(min=1)] = 5,
    split: Annotated[str | None, typer.Option()] = None,
    retry_failed: bool = False,
    video_id: Annotated[list[str] | None, typer.Option("--video-id")] = None,
):
    """生成标注并由模型复核，中断后可继续。"""
    if split not in {None, "train", "val", "test"}:
        raise typer.BadParameter("split 使用 train、val 或 test")
    settings = Settings()
    result = run(Store(settings.root), settings, limit, split, retry_failed, video_id)
    output(result)
    if result["failed"]:
        raise typer.Exit(1)


@app.command()
def status():
    """查看数据量、待复核量和标注来源。"""
    output(Store(Settings().root).stats())


@app.command()
def inspect(video_id: str):
    """查看单个视频的标注、复核来源和调用记录位置。"""
    output(Store(Settings().root).get(video_id))


@app.command()
def requeue(video_ids: Annotated[list[str], typer.Argument()]):
    """更新提示词或模型后，指定视频重新标注；旧版本保留供追溯。"""
    output(Store(Settings().root).requeue(video_ids))


@app.command()
def models():
    """查询标注网关提供的模型名称。"""
    from openai import OpenAI

    settings = Settings()
    if not settings.api_key.get_secret_value():
        raise typer.BadParameter("请配置 VH_DATA_API_KEY")
    with OpenAI(
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        timeout=30,
        max_retries=0,
    ) as client:
        output([model.id for model in client.models.list()])


@app.command("export")
def freeze(destination: Path):
    """冻结模型复核通过的训练、验证和测试标注。"""
    output(export(Store(Settings().root), destination.resolve()))


@app.command()
def feedback(
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], model_id: str
):
    """导入小模型预测，把漏检或意见不同的视频优先交给模型复核。"""
    settings = Settings()
    rows = [
        VideoPredictions.model_validate_json(line)
        for line in predictions.read_text().splitlines()
        if line.strip()
    ]
    output(Store(settings.root).feedback(rows, model_id, settings.agreement_iou))


if __name__ == "__main__":
    app()
