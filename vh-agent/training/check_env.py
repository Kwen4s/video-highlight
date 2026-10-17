"""Check detector training dependencies on CPU, without loading model weights."""

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

from packaging.utils import canonicalize_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data0/wkw/video-highlight"))
    args = parser.parse_args()
    root = args.root.resolve()
    assert Path(sys.prefix).resolve() == root / "envs/train", "请先激活独立训练环境"

    import torch
    from peft import LoraConfig
    from torchcodec.decoders import VideoDecoder
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from trl import SFTTrainer
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config.actor import ActorConfig

    torch.set_num_threads(2)
    assert LoraConfig and SFTTrainer and Qwen3_5ForConditionalGeneration
    for name in ("vllm", "flash_attn", "fla", "ray", "qwen_vl_utils"):
        importlib.import_module(name)
    verl = importlib.import_module("verl")
    assert Path(verl.__file__).resolve().is_relative_to(root / "envs/verl")
    print("训练模块和独立 verl 源码：通过", flush=True)

    model_path = root / "models/Qwen3.5-9B"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert config.model_type == "qwen3_5"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    dataset = root / "datasets/vl-001/sft/train.jsonl"
    with dataset.open() as stream:
        sample = json.loads(next(stream))
    video = sample["video"]
    decoder = VideoDecoder(video["path"], num_ffmpeg_threads=2)
    end = min(video["end_sec"], video["start_sec"] + 2)
    times = torch.linspace(video["start_sec"], end, 5)[:-1].tolist()
    frames = decoder.get_frames_played_at(seconds=times).data
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video["path"]},
                    {"type": "text", "text": "描述这段画面。"},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[text],
        videos=[frames],
        video_metadata=[
            {
                "fps": 4 / (end - video["start_sec"]),
                "total_num_frames": 4,
                "frames_indices": [0, 1, 2, 3],
            }
        ],
        do_sample_frames=False,
        videos_kwargs={"size": {"shortest_edge": 128 * 128, "longest_edge": 4 * 256 * 256}},
        return_tensors="pt",
    )
    assert inputs["pixel_values_videos"].numel() and inputs["input_ids"].numel()
    print("真实短剧视频解码与 Qwen3.5 输入处理：通过", flush=True)

    current = torch.full((4, 6), -1.0, requires_grad=True)
    loss, _ = compute_policy_loss_gspo(
        old_log_prob=current.detach().clone(),
        log_prob=current,
        advantages=torch.tensor([1.0, -1.0, 0.5, -0.5]).unsqueeze(1).expand(4, 6),
        response_mask=torch.ones_like(current),
        config=ActorConfig(strategy="fsdp", rollout_n=4, ppo_micro_batch_size_per_gpu=1),
    )
    loss.backward()
    assert torch.isfinite(loss) and current.grad is not None
    assert torch.isfinite(current.grad).all() and current.grad.abs().sum() > 0
    print("GSPO CPU 损失及反向传播：通过", flush=True)

    packages = {
        canonicalize_name(d.metadata["Name"]): d.version
        for d in importlib.metadata.distributions(path=[sysconfig.get_path("purelib")])
    }
    revision = subprocess.check_output(
        ["git", "-C", str(root / "envs/verl"), "rev-parse", "HEAD"], text=True
    ).strip()
    report = {
        "python": sys.version.split()[0],
        "verl_revision": revision,
        "packages": dict(sorted(packages.items())),
        "video_input_shape": list(inputs["pixel_values_videos"].shape),
        "checks": ["imports", "independent_verl", "real_video_processor", "gspo_cpu_backward"],
    }
    destination = root / "envs/environment.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"版本与检查记录：{destination}")


if __name__ == "__main__":
    main()
