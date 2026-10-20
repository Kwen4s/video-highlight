"""Shared, time-aligned video inputs and answer-only SFT batches."""

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Standalone training entry points use the same source package as the agent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vh_agent.detector_inputs import (  # noqa: F401
    TASK,
    InputConfig,
    VideoInputs,
    dialogue_text,
    input_record,
    messages,
    sample_video,
)


def read_rows(dataset: Path, name: str) -> list[dict]:
    manifest = json.loads((dataset / "manifest.json").read_text())
    path = dataset / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]:
        raise ValueError(f"数据文件指纹不符：{path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class WindowDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class SFTCollator:
    def __init__(self, builder: VideoInputs):
        self.builder = builder

    def __call__(self, rows: list[dict]) -> dict:
        examples = [self.builder(row) for row in rows]
        batch = {}
        for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            if key not in examples[0]:
                continue
            padding = (
                -100
                if key == "labels"
                else (self.builder.processor.tokenizer.pad_token_id if key == "input_ids" else 0)
            )
            batch[key] = pad_sequence(
                [item[key][0] for item in examples], batch_first=True, padding_value=padding
            )
        for key in ("pixel_values_videos", "video_grid_thw"):
            batch[key] = torch.cat([item[key] for item in examples])
        batch.update(answer_positions(batch["labels"]))
        batch["skip_logits"] = True
        return batch


def answer_positions(labels: torch.Tensor) -> dict:
    """Project only positions predicting answers; keep the entire context in the model."""
    shifted = torch.nn.functional.pad(labels[:, 1:], (0, 1), value=-100)
    positions = (shifted != -100).any(dim=0).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("训练批次没有可监督的答案 token")
    return {
        "logits_to_keep": positions,
        "shift_labels": shifted.index_select(1, positions).contiguous(),
    }
