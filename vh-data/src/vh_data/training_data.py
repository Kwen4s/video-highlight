"""Build the sole SFT/GSPO dataset with frozen full-video references."""

import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exporter import export
from .models import Annotation, Span, digest, overlap
from .quality import subtract, union
from .sampling import select_gspo, stage_statistics
from .store import ConflictError, Store

SPLITS = ("train", "val", "test")


class VideoWindow(Span):
    path: str = Field(description="原视频绝对路径")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$", description="原视频内容指纹")


class WindowRange(Span):
    start_sec: float = Field(ge=0, description="观看窗口内的开始秒数")
    end_sec: float = Field(gt=0, description="观看窗口内的结束秒数")


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start_sec: float = Field(description="观看窗口相对开始秒数；负数表示前情")
    end_sec: float = Field(description="观看窗口相对结束秒数")
    text: str
    source: Literal["asr", "ocr"]

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("对白结束时间必须晚于开始时间")
        return self


class VLWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    video_id: str
    drama_id: str
    label_id: int
    video: VideoWindow
    detect_range: WindowRange
    dialogue: list[Dialogue]
    segments: list[WindowRange]

    @model_validator(mode="after")
    def timeline(self):
        duration = self.video.end_sec - self.video.start_sec
        if self.detect_range.end_sec > duration + 1e-6:
            raise ValueError("检测范围超出观看范围")
        for segment in self.segments:
            midpoint = (segment.start_sec + segment.end_sec) / 2
            if segment.end_sec > duration + 1e-6 or not (
                self.detect_range.start_sec <= midpoint < self.detect_range.end_sec
            ):
                raise ValueError("高光边界或窗口归属无效")
        if self.segments != sorted(self.segments, key=lambda s: s.start_sec):
            raise ValueError("高光必须按时间排序")
        if any(overlap(a, b) for a, b in zip(self.segments, self.segments[1:], strict=False)):
            raise ValueError("高光不能重叠")
        if any(t.end_sec > duration + 1e-6 for t in self.dialogue):
            raise ValueError("对白包含观看范围之后的信息")
        return self


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative(span: Span, origin: float) -> WindowRange:
    return WindowRange(
        start_sec=round(span.start_sec - origin, 6),
        end_sec=round(span.end_sec - origin, 6),
    )


def build_windows(
    video: dict,
    version: dict,
    dialogue: list[Dialogue],
    *,
    window_sec: float,
    context_sec: float,
    dialogue_history_sec: float,
) -> tuple[list[VLWindow], Counter]:
    annotation = Annotation.model_validate(version["annotation"])
    positives = sorted(
        (s for s in annotation.segments if s.kind == "positive"), key=lambda s: s.start_sec
    )
    known = union([s for s in annotation.segments if s.kind != "uncertain"])
    windows, counts = [], Counter()
    for index in range(math.ceil(annotation.duration_sec / window_sec)):
        core = Span(
            start_sec=index * window_sec,
            end_sec=min((index + 1) * window_sec, annotation.duration_sec),
        )
        counts["total"] += 1
        if sum(s.end_sec - s.start_sec for s in subtract(core, known)) > 1e-6:
            counts["unknown"] += 1
            continue
        read = Span(
            start_sec=max(0, core.start_sec - context_sec),
            end_sec=min(annotation.duration_sec, core.end_sec + context_sec),
        )
        targets = [
            s for s in positives if core.start_sec <= (s.start_sec + s.end_sec) / 2 < core.end_sec
        ]
        if any(s.start_sec < read.start_sec or s.end_sec > read.end_sec for s in targets):
            counts["incomplete_core"] += 1
            continue
        history_start = max(0, read.start_sec - dialogue_history_sec)
        text = [
            Dialogue(
                start_sec=round(t.start_sec - read.start_sec, 6),
                end_sec=round(t.end_sec - read.start_sec, 6),
                text=t.text,
                source=t.source,
            )
            for t in dialogue
            if history_start <= t.start_sec < t.end_sec <= read.end_sec
        ]
        windows.append(
            VLWindow(
                sample_id=f"{video['video_id']}:{core.start_sec:g}-{core.end_sec:g}",
                video_id=video["video_id"],
                drama_id=video["drama_id"],
                label_id=version["label_id"],
                video=VideoWindow(path=video["path"], sha256=video["sha256"], **read.model_dump()),
                detect_range=_relative(core, read.start_sec),
                dialogue=text,
                segments=[_relative(s, read.start_sec) for s in targets],
            )
        )
        counts["usable"] += 1
        counts["positive" if targets else "empty"] += 1
    return windows, counts


def _history_windows(root: Path, version: dict, windows: list[VLWindow]):
    annotation = Annotation.model_validate(version["annotation"])
    negative = union([s for s in annotation.segments if s.kind == "negative"])
    positive = [s for s in annotation.segments if s.kind == "positive"]
    provenance = version["provenance"]
    reviews = set(provenance.get("review_calls", []))
    candidates = []
    for call in sorted(set(provenance.get("calls", [])) - reviews):
        record = json.loads((root / call).read_text())
        if record["input"]["pass"] != "scan":
            continue
        for item in record["result"]["segments"]:
            if item["kind"] != "positive":
                continue
            span = Span.model_validate({k: item[k] for k in ("start_sec", "end_sec")})
            if sum(s.end_sec - s.start_sec for s in subtract(span, negative)) <= 1e-6 and not any(
                overlap(span, s) for s in positive
            ):
                candidates.append(span)
    regions = union(candidates)
    focus_ids = {
        w.sample_id
        for w in windows
        if any(
            overlap(
                span,
                Span(
                    start_sec=w.video.start_sec + w.detect_range.start_sec,
                    end_sec=w.video.start_sec + w.detect_range.end_sec,
                ),
            )
            for span in regions
        )
    }
    return len(regions), focus_ids


def prepare_training(
    store: Store,
    destination: Path,
    *,
    window_sec: float = 20,
    context_sec: float = 15,
    dialogue_history_sec: float = 90,
    gspo_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Publish one dataset: SFT, GSPO, shared evaluation and source provenance."""
    for name, value in (
        ("window_sec", window_sec),
        ("context_sec", context_sec),
        ("dialogue_history_sec", dialogue_history_sec),
    ):
        if not math.isfinite(value) or value < 0 or (name == "window_sec" and value == 0):
            raise ValueError(f"{name} 的值无效")
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError("数据目录已存在，请使用新的版本名称")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".training-", dir=destination.parent))
    try:
        source_report = export(store, temporary / "source")
        videos = [
            json.loads(line)
            for line in (temporary / "source/annotations.jsonl").read_text().splitlines()
        ]
        versions = {
            v["video_id"]: v
            for v in json.loads((temporary / "source/provenance.json").read_text())["versions"]
        }
        paths = {"train": "sft/train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
        for stage in ("sft", "gspo"):
            (temporary / stage).mkdir()
        for name in [*paths.values(), "source/dialogues.jsonl"]:
            (temporary / name).touch()
        counts = {split: Counter() for split in SPLITS}
        groups, train, focus_ids, history_regions = {}, [], set(), 0
        for video in videos:
            split, drama = video["split"], video["drama_id"]
            if split not in SPLITS or (drama in groups and groups[drama] != split):
                raise ConflictError("短剧划分无效或跨训练/验证/测试分组")
            groups[drama] = split
            version = versions[video["video_id"]]
            record = json.loads((store.root / version["provenance"]["dialogue"]).read_text())
            if (
                record["input"]["source_sha256"] != video["sha256"]
                or record["input"]["source_start_time_sec"] != 0
            ):
                raise ConflictError(f"{video['video_id']} 的对白未对齐源视频")
            dialogue = sorted(
                (Dialogue.model_validate(t) for t in record["segments"]),
                key=lambda t: (t.start_sec, t.end_sec, t.source),
            )
            if any(t.start_sec < 0 or t.end_sec > video["duration_sec"] + 1e-6 for t in dialogue):
                raise ConflictError(f"{video['video_id']} 的对白越界")
            with (temporary / "source/dialogues.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "video_id": video["video_id"],
                            "segments": [t.model_dump() for t in dialogue],
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            windows, statistics = build_windows(
                video,
                version,
                dialogue,
                window_sec=window_sec,
                context_sec=context_sec,
                dialogue_history_sec=dialogue_history_sec,
            )
            with (temporary / paths[split]).open("a", encoding="utf-8") as stream:
                for window in windows:
                    stream.write(
                        json.dumps(window.model_dump(), ensure_ascii=False, allow_nan=False) + "\n"
                    )
            counts[split].update(statistics)
            if split == "train":
                train.extend(windows)
                regions, linked = _history_windows(store.root, version, windows)
                history_regions += regions
                focus_ids.update(linked)
        if gspo_size is None:
            if not focus_ids:
                raise ValueError("没有历史采样线索，请明确指定 --gspo-size")
            gspo_size = min(len(train), 2 * len(focus_ids))
        selected = select_gspo(train, focus_ids, gspo_size, seed)
        with (temporary / "gspo/train.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(row.model_dump_json() + "\n")
        _write_json(temporary / "schema.json", VLWindow.model_json_schema())
        report = {
            "format": "vh-vl-training-v1",
            "sample_format": "vh-vl-window-v1",
            "source": source_report,
            "window_sec": window_sec,
            "context_sec": context_sec,
            "dialogue_history_sec": dialogue_history_sec,
            "seed": seed,
            "time_reference": "video: original seconds; other times: relative to video.start_sec",
            "sft": stage_statistics(train, focus_ids),
            "gspo": stage_statistics(selected, focus_ids),
            "window_counts": {split: dict(counts[split]) for split in SPLITS},
            "sampling": {
                "size_rule": "explicit size, or min(train size, 2 * linked history windows)",
                "strata": "empty/single/multiple proportional to SFT",
                "within_stratum": "include history windows; draw the rest by drama round-robin",
                "focus_sample_ids": sorted(focus_ids),
                "history_regions": history_regions,
            },
            "sft_gspo_overlap": len(selected),
            "answer_usage": {
                "sft": "segments become the assistant target",
                "gspo": "segments go only to the reward function",
            },
            "evaluation": {
                "val": counts["val"]["usable"],
                "test": counts["test"]["usable"],
                "full_video_source": "source",
            },
            "files": {
                name: digest(temporary / name)
                for name in [
                    "sft/train.jsonl",
                    "gspo/train.jsonl",
                    "val.jsonl",
                    "test.jsonl",
                    "schema.json",
                    "source/annotations.jsonl",
                    "source/provenance.json",
                    "source/manifest.json",
                    "source/dialogues.jsonl",
                ]
            },
        }
        _write_json(temporary / "manifest.json", report)
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
