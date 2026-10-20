"""Canonical detector inputs, shared by SFT, GSPO and the local inference service."""

import json
import math
from dataclasses import asdict, dataclass

import torch
from torchcodec.decoders import VideoDecoder

TASK = (
    "结合画面和对白，找出值得单独回看的精彩核心，圈住精彩实际发生的台词、动作或反应。"
    "普通铺垫和重复内容留作理解前情。片段通常几秒，优先15秒以内。"
    "只报告核心中点位于本次检测范围的片段，保留完整起止，按时间排序。"
    '时间使用当前视频窗口的秒数，返回JSON：{"segments":[{"start_sec":数值,"end_sec":数值}]}。'
    '没有高光时返回{"segments":[]}。'
)


@dataclass(frozen=True)
class InputConfig:
    fps: float = 2
    visual_tokens: int = 8192
    max_length: int = 16384


def dialogue_text(row: dict) -> str:
    # Previous spoken dialogue supplies the story; screen text belongs to visible frames.
    lines = []
    for item in row["dialogue"]:
        if item["source"] == "ocr" and item["end_sec"] <= 0:
            continue
        start = max(0, item["start_sec"]) if item["source"] == "ocr" else item["start_sec"]
        kind = "对白" if item["source"] == "asr" else "画面文字"
        lines.append(f"{start:.2f}–{item['end_sec']:.2f} {kind}：{item['text']}")
    return "\n".join(lines)


def messages(row: dict) -> list[dict]:
    core = row["detect_range"]
    text = (
        f"本次检测范围：{core['start_sec']:.2f}–{core['end_sec']:.2f}秒。"
        "下列文字使用同一时间轴，负时间是此前对白。\n" + dialogue_text(row)
    )
    return [
        {"role": "system", "content": TASK},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": row["video"]["path"]},
                {"type": "text", "text": text},
            ],
        },
    ]


def sample_video(video: dict, fps: float):
    decoder = VideoDecoder(video["path"], num_ffmpeg_threads=2)
    # Stream bounds are authoritative; container duration may include a longer audio tail.
    end = min(video["end_sec"], decoder.metadata.end_stream_seconds)
    if not decoder.metadata.begin_stream_seconds <= video["start_sec"] < end:
        raise ValueError("观看范围不在实际视频流内")
    duration = end - video["start_sec"]
    count = max(1, math.ceil(duration * fps))
    times = video["start_sec"] + torch.arange(count, dtype=torch.float64) / fps
    times = times[times < end]
    count = times.numel()
    frames = decoder.get_frames_played_at(seconds=times.tolist()).data
    return frames, {"fps": fps, "total_num_frames": count, "frames_indices": list(range(count))}


class VideoInputs:
    def __init__(self, processor, config: InputConfig):
        self.processor = processor
        self.config = config

    def __call__(self, row: dict, *, answer: bool = True) -> dict:
        frames, video_metadata = sample_video(row["video"], self.config.fps)
        template = self.processor.apply_chat_template(
            messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        vp = self.processor.video_processor
        pixel_budget = (
            self.config.visual_tokens * vp.patch_size**2 * vp.merge_size**2 * vp.temporal_patch_size
        )
        inputs = self.processor(
            text=[template],
            videos=[frames],
            video_metadata=[video_metadata],
            do_sample_frames=False,
            videos_kwargs={
                "size": {"shortest_edge": vp.size["shortest_edge"], "longest_edge": pixel_budget}
            },
            return_tensors="pt",
        )
        prefix_length = inputs["input_ids"].shape[-1]
        result = {key: value for key, value in inputs.items()}
        if answer:
            target = json.dumps(
                {"segments": row["segments"]}, ensure_ascii=False, separators=(",", ":")
            )
            tokens = self.processor.tokenizer.encode(target, add_special_tokens=False)
            tokens.append(self.processor.tokenizer.eos_token_id)
            suffix = torch.tensor([tokens], dtype=torch.long)
            result["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
            result["attention_mask"] = torch.ones_like(result["input_ids"])
            if "mm_token_type_ids" in result:
                result["mm_token_type_ids"] = torch.cat(
                    [result["mm_token_type_ids"], torch.zeros_like(suffix)], dim=1
                )
            result["labels"] = result["input_ids"].clone()
            result["labels"][:, :prefix_length] = -100
        if result["input_ids"].shape[-1] > self.config.max_length:
            raise ValueError(f"{row['sample_id']} 超出序列预算，不能截断视频或答案")
        return result


def input_record(config: InputConfig) -> dict:
    return {
        **asdict(config),
        "task": TASK,
        "dialogue": "previous and current ASR; OCR from visible video only",
    }
