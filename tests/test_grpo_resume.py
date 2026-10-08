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

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer

from trl import GRPOTrainer


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
