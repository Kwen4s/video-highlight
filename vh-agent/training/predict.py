"""Generate highlight intervals from a frozen adapter, without reference answers."""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import torch
from accelerate import PartialState
from data import InputConfig, VideoInputs, WindowDataset, input_record, read_rows
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class PromptInputs:
    def __init__(self, builder):
        self.builder = builder

    def __call__(self, row):
        prompt = {key: value for key, value in row.items() if key != "segments"}
        return row["sample_id"], self.builder(prompt, answer=False)


def committed_records(stream):
    """Discard only an interrupted final write before resuming the same journal."""
    stream.seek(0)
    records = []
    end = 0
    while line := stream.readline():
        if not line.endswith("\n"):
            stream.seek(end)
            stream.truncate()
            break
        records.append(json.loads(line))
        end = stream.tell()
    stream.seek(0, 2)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="已完成训练的目录")
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "completed.json").is_file():
        parser.error("训练尚未完成")
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_num_threads(4)
    distributed = PartialState()
    metadata = json.loads((output / "run.json").read_text())
    dataset = Path(metadata["dataset"])
    rows = read_rows(dataset, "test.jsonl")
    config = InputConfig(
        **{key: metadata["input"][key] for key in ("fps", "visual_tokens", "max_length")}
    )
    if input_record(config) != metadata["input"]:
        raise ValueError("推理输入与训练记录不一致")
    manifest = json.loads((dataset / "manifest.json").read_text())["files"]
    if manifest["test.jsonl"] != metadata["dataset_manifest"]["test.jsonl"]:
        raise ValueError("测试集已改变")
    test_dramas = {row["drama_id"] for row in rows}
    for name in ("sft/train.jsonl", "val.jsonl"):
        if test_dramas & {row["drama_id"] for row in read_rows(dataset, name)}:
            raise ValueError("测试剧与训练或验证交叉")
    folder = output / "effect"
    if distributed.is_main_process:
        folder.mkdir(exist_ok=True)
        with (output / "adapter/adapter_model.safetensors").open("rb") as stream:
            adapter_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        protocol = {
            "adapter_sha256": adapter_hash,
            "dataset_sha256": manifest["test.jsonl"],
            "input": metadata["input"],
            "generation": {"do_sample": False, "max_new_tokens": 512, "use_cache": True},
            "samples": len(rows),
            "world_size": distributed.num_processes,
        }
        path = folder / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("已有预测的权重、数据或生成配置不一致")
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    distributed.wait_for_everyone()
    protocol = json.loads((folder / "protocol.json").read_text())
    rank = distributed.process_index
    selected = rows[rank :: distributed.num_processes]
    with (folder / f".rank-{rank}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder / f"predictions-{rank}.jsonl").open("a+") as journal:
            records = committed_records(journal)
            completed = {item["sample_id"] for item in records}
            pending = [row for row in selected if row["sample_id"] not in completed]
            if pending:
                processor = AutoProcessor.from_pretrained(output / "adapter", local_files_only=True)
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    output / "adapter",
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    local_files_only=True,
                    device_map={"": distributed.device},
                ).eval()
                if not model.active_adapters():
                    raise ValueError("未加载最终 LoRA")
                loader = DataLoader(
                    WindowDataset(pending),
                    batch_size=None,
                    num_workers=2,
                    collate_fn=PromptInputs(VideoInputs(processor, config)),
                    prefetch_factor=2,
                    multiprocessing_context="spawn",
                )
                for sample_id, inputs in loader:
                    inputs = {key: value.to(distributed.device) for key, value in inputs.items()}
                    prefix = inputs["input_ids"].shape[-1]
                    if prefix + protocol["generation"]["max_new_tokens"] > config.max_length:
                        raise ValueError(f"{sample_id} 生成空间不足，不截断输入")
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            **protocol["generation"],
                            temperature=None,
                            top_p=None,
                            top_k=None,
                        )[0, prefix:]
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos
                    item = {
                        "sample_id": sample_id,
                        "text": processor.tokenizer.decode(generated, skip_special_tokens=True),
                        "input_tokens": prefix,
                        "output_tokens": generated.numel(),
                        "generation_sec": seconds,
                        "truncated": generated.numel() == protocol["generation"]["max_new_tokens"]
                        and generated[-1].item() not in eos,
                    }
                    journal.write(json.dumps(item, ensure_ascii=False) + "\n")
                    journal.flush()
                    completed.add(sample_id)
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "done": len(completed),
                                "total": len(selected),
                                "last": item,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    distributed.wait_for_everyone()
    if distributed.is_main_process:
        from score import build_report

        build_report(output)
    distributed.wait_for_everyone()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
