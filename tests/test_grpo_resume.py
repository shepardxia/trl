# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer

from trl import GRPOConfig, GRPOTrainer


def bare_trainer(**attributes):
    """A `GRPOTrainer` holding only what the method under test reads."""
    trainer = object.__new__(GRPOTrainer)
    trainer.__dict__.update(attributes)
    return trainer


def adapter_weights(model, adapter):
    return {n: p.detach().clone() for n, p in model.named_parameters() if f".{adapter}." in n}


@pytest.fixture
def checkpoint(tmp_path):
    """A model whose trained "default" adapter sits beside a "ref" one, as saved, then perturbed in memory."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8,
    )  # fmt: skip
    lora = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False, task_type="CAUSAL_LM")
    model = get_peft_model(Qwen3ForCausalLM(config), lora)
    model.add_adapter("ref", lora)
    model.set_adapter("default")
    model.save_pretrained(tmp_path)
    saved = {name: adapter_weights(model, name) for name in ("default", "ref")}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(1.0)
    return model, saved, str(tmp_path)


def test_parent_resume_leaves_the_trained_adapter_unloaded(checkpoint):
    model, saved, path = checkpoint
    Trainer._load_from_checkpoint(bare_trainer(model=model), path)
    for name, weight in adapter_weights(model, "ref").items():
        torch.testing.assert_close(weight, saved["ref"][name])
    assert all(not torch.equal(w, saved["default"][n]) for n, w in adapter_weights(model, "default").items())


def test_resume_loads_the_trained_adapter(checkpoint):
    model, saved, path = checkpoint
    bare_trainer(model=model)._load_from_checkpoint(path)
    for adapter in ("default", "ref"):
        for name, weight in adapter_weights(model, adapter).items():
            torch.testing.assert_close(weight, saved[adapter][name])
    assert model.active_adapters == ["default"]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".default." in n for n in trainable)


def learning_rates(args, steps):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = bare_trainer(args=args, optimizer=optimizer, lr_scheduler=None).create_scheduler(steps)
    rates = []
    for _ in range(steps + 1):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return rates


def test_warmup_starts_at_the_given_rate(tmp_path):
    args = GRPOConfig(
        output_dir=str(tmp_path),
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.2},
        warmup_steps=10,
        warmup_start_rate=0.1,
    )
    rates = learning_rates(args, steps=50)
    assert rates[0] == pytest.approx(0.1)
    assert rates[5] == pytest.approx(0.55)
    assert rates[10] == pytest.approx(1.0)
    assert rates[30] == pytest.approx(0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * 0.5)))
    assert rates[50] == pytest.approx(0.2)


@pytest.mark.parametrize("schedule", ["constant_with_warmup", "linear", "cosine", "cosine_with_min_lr"])
def test_schedule_after_the_warmup_is_the_stock_one(tmp_path, schedule):
    kwargs = {"min_lr_rate": 0.2} if schedule == "cosine_with_min_lr" else {}
    args = GRPOConfig(
        output_dir=str(tmp_path), lr_scheduler_type=schedule, lr_scheduler_kwargs=kwargs, warmup_steps=10
    )
    stock = learning_rates(args, steps=50)
    args.warmup_start_rate = 0.5
    ramped = learning_rates(args, steps=50)
    assert ramped[0] == pytest.approx(0.5)
    assert ramped[10:] == pytest.approx(stock[10:])


def test_training_resumes_a_warm_started_adapter(tmp_path, monkeypatch):
    """Two packed steps of two microbatches on a pretrained adapter with a KL term, then a resume from the
    last checkpoint."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM

    from trl.trainer import prefix_pack

    from .test_prefix_pack import dense_varlen

    monkeypatch.setattr(prefix_pack, "_varlen", dense_varlen)
    monkeypatch.setattr(prefix_pack, "_stock_attn", None)
    packed_hidden, packed_calls = prefix_pack.packed_hidden, []
    monkeypatch.setattr(
        prefix_pack, "packed_hidden", lambda *args: packed_calls.append(None) or packed_hidden(*args)
    )
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    def trainer():
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        lora = LoraConfig(target_modules=["q_proj", "v_proj"], init_lora_weights=False, task_type="CAUSAL_LM")
        args = GRPOConfig(
            output_dir=str(tmp_path),
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            gradient_accumulation_steps=2,
            max_completion_length=8,
            beta=0.1,
            prefix_pack=True,
            warmup_steps=1,
            warmup_start_rate=0.5,
            max_steps=2,
            save_steps=1,
            report_to="none",
        )
        return GRPOTrainer(
            model=get_peft_model(model, lora),
            reward_funcs=lambda completions, **kwargs: [float(len(c)) for c in completions],
            args=args,
            train_dataset=dataset,
        )

    first = trainer()
    initial = adapter_weights(first.model, "default")
    first.train()
    assert first.prefix_pack and "ref" in first.model.peft_config
    assert len(packed_calls) == 4  # every microbatch holds one prompt, since the generation batch is not shuffled
    trained = adapter_weights(first.model, "default")
    assert any(not torch.equal(w, initial[n]) for n, w in trained.items())
    for name, weight in adapter_weights(first.model, "ref").items():
        torch.testing.assert_close(weight, initial[name.replace(".ref.", ".default.")])

    resumed = trainer()
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    for name, weight in adapter_weights(resumed.model, "default").items():
        torch.testing.assert_close(weight, trained[name])
