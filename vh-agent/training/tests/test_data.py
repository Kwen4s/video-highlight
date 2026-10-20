import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import InputConfig, SFTCollator, VideoInputs, answer_positions, messages, read_rows

ROOT = Path("/data0/wkw/video-highlight")


@pytest.fixture(scope="module")
def processor():
    torch.set_num_threads(2)
    return AutoProcessor.from_pretrained(ROOT / "models/Qwen3.5-9B", local_files_only=True)


@pytest.fixture
def row():
    sample = copy.deepcopy(read_rows(ROOT / "datasets/vl-001", "sft/train.jsonl")[0])
    sample["video"]["start_sec"] = 0
    sample["video"]["end_sec"] = 2
    sample["detect_range"] = {"start_sec": 0, "end_sec": 2}
    sample["dialogue"] = [
        {"start_sec": -2, "end_sec": -1, "source": "asr", "text": "前面的关系"},
        {"start_sec": -2, "end_sec": -1, "source": "ocr", "text": "历史画面水印"},
        {"start_sec": 0, "end_sec": 1, "source": "ocr", "text": "当前证据"},
    ]
    sample["segments"] = [{"start_sec": 0.5, "end_sec": 1.25}]
    return sample


def test_story_uses_spoken_history_and_visible_screen_text(row):
    prompt = json.dumps(messages(row), ensure_ascii=False)
    assert "前面的关系" in prompt and "-2.00" in prompt and "当前证据" in prompt
    assert "历史画面水印" not in prompt
    assert "segments" not in messages(row)[1]["content"][1]["text"]


def test_real_video_answer_mask_and_no_reference_leakage(processor, row):
    builder = VideoInputs(processor, InputConfig(visual_tokens=128))
    first = builder(row)
    prefix = (first["labels"][0] == -100).sum().item()
    target = processor.tokenizer.decode(first["labels"][0, prefix:], skip_special_tokens=True)
    assert json.loads(target) == {"segments": row["segments"]}
    other = copy.deepcopy(row)
    other["segments"] = []
    second = builder(other)
    assert torch.equal(first["input_ids"][:, :prefix], second["input_ids"][:, :prefix])
    assert first["pixel_values_videos"].shape == second["pixel_values_videos"].shape
    assert (first["labels"][0, :prefix] == -100).all()
    assert first["video_grid_thw"][0, 0] == 2  # 2 seconds, 2 FPS, temporal patch size 2.


def test_batch_padding_does_not_supervise_or_mark_video_tokens(processor, row):
    other = copy.deepcopy(row)
    other["segments"] = []
    batch = SFTCollator(VideoInputs(processor, InputConfig(visual_tokens=128)))([row, other])
    padding = batch["attention_mask"] == 0
    assert padding.any() and (batch["labels"][padding] == -100).all()
    if "mm_token_type_ids" in batch:
        assert (batch["mm_token_type_ids"][padding] == 0).all()
    assert batch["video_grid_thw"].shape[0] == 2
    expected = torch.nn.functional.pad(batch["labels"][:, 1:], (0, 1), value=-100)
    assert torch.equal(batch["shift_labels"], expected[:, batch["logits_to_keep"]])
    assert (batch["shift_labels"] != -100).sum() == (expected != -100).sum()


def test_over_budget_input_raises_instead_of_truncating(processor, row):
    with pytest.raises(ValueError, match="不能截断"):
        VideoInputs(processor, InputConfig(visual_tokens=128, max_length=10))(row)


@pytest.mark.parametrize("tail_extension", [0, 1e-6, 0.25])
def test_real_stream_end_is_exclusive_and_labels_remain_unchanged(processor, row, tail_extension):
    end = VideoDecoder(row["video"]["path"]).metadata.end_stream_seconds
    row["video"].update(start_sec=end - 1, end_sec=end + tail_extension)
    row["detect_range"] = {"start_sec": 0, "end_sec": 1}
    row["segments"] = [{"start_sec": 0.2, "end_sec": 0.7}]
    inputs = VideoInputs(processor, InputConfig(visual_tokens=128))(row)
    # Exactly two real frames at 2 FPS, regardless of a nominal tail past EOF.
    assert inputs["video_grid_thw"][0, 0] == 1
    target = inputs["labels"][inputs["labels"] != -100]
    assert json.loads(processor.tokenizer.decode(target, skip_special_tokens=True)) == {
        "segments": row["segments"]
    }
    row["video"]["start_sec"] = end
    with pytest.raises(ValueError, match="实际视频流"):
        VideoInputs(processor, InputConfig(visual_tokens=128))(row)


def test_answer_positions_preserve_causal_loss_and_all_gradients():
    torch.manual_seed(42)
    labels = torch.tensor([[-100, -100, -100, 3, 5, -100, -100], [-100, -100, 2, -100, 4, 6, -100]])
    hidden = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    head = torch.randn(8, 12, dtype=torch.float64, requires_grad=True)
    shifted = torch.nn.functional.pad(labels[:, 1:], (0, 1), value=-100)
    full = torch.nn.functional.cross_entropy((hidden @ head.T).flatten(0, 1), shifted.flatten())
    full_gradients = torch.autograd.grad(full, (hidden, head))
    selected = answer_positions(labels)
    logits = hidden[:, selected["logits_to_keep"]] @ head.T
    compact = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), selected["shift_labels"].flatten()
    )
    compact_gradients = torch.autograd.grad(compact, (hidden, head))
    torch.testing.assert_close(full, compact, rtol=1e-12, atol=1e-12)
    for full_gradient, compact_gradient in zip(full_gradients, compact_gradients, strict=True):
        torch.testing.assert_close(full_gradient, compact_gradient, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="没有可监督"):
        answer_positions(torch.full((1, 5), -100))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 验证实际融合损失")
def test_liger_answer_projection_preserves_loss_and_hidden_gradient():
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

    torch.manual_seed(42)
    hidden = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    head = torch.randn(1024, 128, device="cuda", dtype=torch.bfloat16) / 128**0.5
    labels = torch.full((2, 64), -100, device="cuda", dtype=torch.long)
    labels[0, 40:55] = torch.arange(15, device="cuda")
    labels[1, 50:60] = torch.arange(10, device="cuda")
    full = LigerForCausalLMLoss(hidden, head, labels, hidden_size=128)
    full_gradient = torch.autograd.grad(full, hidden)[0]
    selected = answer_positions(labels)
    compact = LigerForCausalLMLoss(
        hidden[:, selected["logits_to_keep"]].contiguous(),
        head,
        labels,
        hidden_size=128,
        shift_labels=selected["shift_labels"],
    )
    compact_gradient = torch.autograd.grad(compact, hidden)[0]
    torch.testing.assert_close(full, compact, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(full_gradient, compact_gradient, rtol=0.02, atol=2e-5)
