"""Serve the trained detector on localhost in its existing inference environment."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .detector_inputs import InputConfig, VideoInputs, input_record


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    read_start_sec: float = Field(ge=0)
    read_end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def valid(self):
        if not self.read_start_sec <= self.start_sec < self.end_sec <= self.read_end_sec:
            raise ValueError("观看范围必须完整包含负责范围")
        if self.end_sec - self.start_sec > 20 + 1e-6:
            raise ValueError("单次检测负责范围最多 20 秒")
        if (
            self.start_sec - self.read_start_sec > 15 + 1e-6
            or self.read_end_sec - self.end_sec > 15 + 1e-6
        ):
            raise ValueError("检测上下文每侧最多 15 秒")
        return self


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    start_sec: float
    end_sec: float
    text: str
    source: str

    @model_validator(mode="after")
    def valid(self):
        if self.end_sec <= self.start_sec or self.source not in {"asr", "ocr"}:
            raise ValueError("对白范围或来源无效")
        return self


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_path: Path
    source_start_time_sec: float
    window: Window
    dialogue: list[Dialogue]


def create_app(adapter, device):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    adapter = Path(adapter).resolve()
    weights = adapter / "adapter_model.safetensors"
    with weights.open("rb") as stream:
        revision = hashlib.file_digest(stream, "sha256").hexdigest()
    config = InputConfig()
    processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        adapter,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        device_map={"": device},
    ).eval()
    if not model.active_adapters():
        raise ValueError("检测服务未加载 LoRA 权重")
    builder = VideoInputs(processor, config)
    profile = {
        "revision": revision,
        "input": input_record(config),
        "window_sec": 20,
        "context_sec": 15,
    }
    lock = Lock()
    app = FastAPI(title="Video highlight detector", docs_url=None, redoc_url=None)

    @app.get("/profile")
    def get_profile():
        return profile

    @app.post("/detect")
    def detect(request: Request):
        window = request.window
        origin = request.source_start_time_sec
        row = {
            "sample_id": f"{window.start_sec:g}-{window.end_sec:g}",
            "video": {
                "path": str(request.video_path),
                "start_sec": window.read_start_sec + origin,
                "end_sec": window.read_end_sec + origin,
            },
            "detect_range": {
                "start_sec": window.start_sec - window.read_start_sec,
                "end_sec": window.end_sec - window.read_start_sec,
            },
            "dialogue": [d.model_dump() for d in request.dialogue],
        }
        try:
            with lock, torch.inference_mode():
                inputs = {k: v.to(device) for k, v in builder(row, answer=False).items()}
                prefix = inputs["input_ids"].shape[-1]
                if prefix + 512 > config.max_length:
                    raise ValueError("输入超出生成预算")
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                tokens = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=512,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                )[0, prefix:]
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            if len(tokens) == 512 and tokens[-1].item() not in eos:
                raise ValueError("模型输出触及长度上限")
            parsed = json.loads(raw)
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"segments"}
                or not isinstance(parsed["segments"], list)
            ):
                raise ValueError("检测输出不是 segments 列表")
            segments, invalid = [], []
            duration = window.read_end_sec - window.read_start_sec
            for s in parsed["segments"]:
                valid = isinstance(s, dict) and set(s) == {"start_sec", "end_sec"}
                values = [s.get("start_sec"), s.get("end_sec")] if isinstance(s, dict) else []
                valid = valid and all(type(v) in (int, float) and math.isfinite(v) for v in values)
                if valid:
                    start, end = values
                    midpoint = (start + end) / 2 + window.read_start_sec
                    valid = (
                        0 <= start < end <= duration
                        and window.start_sec <= midpoint < window.end_sec
                    )
                if valid:
                    segments.append(
                        {
                            "start_sec": start + window.read_start_sec,
                            "end_sec": end + window.read_start_sec,
                        }
                    )
                else:
                    invalid.append(repr(s))
            return {
                "segments": segments,
                "invalid_segments": invalid,
                "raw_output": raw,
                "generation_sec": seconds,
                "model_revision": revision,
            }
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("/data0/wkw/video-highlight/runs/qwen35-9b-gspo-001/adapter"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    uvicorn.run(create_app(args.adapter, args.device), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
