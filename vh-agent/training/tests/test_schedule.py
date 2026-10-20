"""Verify periodic and final evaluation across a checkpoint policy change."""

import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sft import TrainingSchedule


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, input_values, labels):
        logits = self.layer(input_values)
        return {"loss": (logits - labels).square().mean(), "logits": logits}


class Evaluations(TrainerCallback):
    def __init__(self):
        self.steps = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
        self.steps.append(state.global_step)


def test_resume_updates_schedule_and_retains_one_checkpoint(tmp_path):
    samples = [{"input_values": torch.ones(2), "labels": torch.zeros(1)}] * 8

    def arguments(**kwargs):
        return TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            report_to="none",
            disable_tqdm=True,
            save_total_limit=1,
            **kwargs,
        )

    old = Trainer(
        model=TinyModel(),
        args=arguments(max_steps=49, save_steps=49, eval_strategy="no", eval_steps=500),
        train_dataset=samples,
    )
    old.train()
    checkpoint = tmp_path / "checkpoint-49"
    assert checkpoint.is_dir()
    assert old.state.eval_steps == 500
    evaluations = Evaluations()
    trainer = Trainer(
        model=TinyModel(),
        args=arguments(max_steps=51, save_steps=50, eval_steps=50, eval_strategy="steps"),
        train_dataset=samples,
        eval_dataset=samples,
        callbacks=[TrainingSchedule(), evaluations],
    )
    trainer.train(resume_from_checkpoint=str(checkpoint))
    assert trainer.state.eval_steps == trainer.state.save_steps == 50
    assert evaluations.steps == [50, 51]  # periodic and final, without duplication
    assert [p.name for p in tmp_path.glob("checkpoint-*")] == ["checkpoint-51"]
