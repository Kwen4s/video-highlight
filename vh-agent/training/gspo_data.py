"""verl dataset adapter using exactly the same frames and prompts as SFT."""

import asyncio
import json
from pathlib import Path

import torch
from data import messages, read_rows, sample_video
from torch.utils.data import Dataset


class HighlightDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        files = [data_files] if isinstance(data_files, str) else list(data_files)
        self.rows = []
        for file in files:
            path = Path(file)
            root = path.parent.parent if path.parent.name == "gspo" else path.parent
            self.rows.extend(read_rows(root, str(path.relative_to(root))))
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.config = config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = messages(row)
        prompt[1]["content"][0].update(
            video_start_sec=row["video"]["start_sec"], video_end_sec=row["video"]["end_sec"]
        )
        truth = {
            "sample_id": row["sample_id"],
            "segments": row["segments"],
            "duration_sec": row["video"]["end_sec"] - row["video"]["start_sec"],
            "detect_range": row["detect_range"],
        }
        return {
            "index": index,
            "raw_prompt": prompt,
            "data_source": "short_drama",
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(truth, ensure_ascii=False),
            },
            "extra_info": {"index": index, "sample_id": row["sample_id"]},
        }

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        item = messages[1]["content"][0]
        video = {
            "path": item["video"],
            "start_sec": item["video_start_sec"],
            "end_sec": item["video_end_sec"],
        }
        frames, metadata = await asyncio.to_thread(sample_video, video, config.video_fps)
        return None, [(frames, metadata)], None
