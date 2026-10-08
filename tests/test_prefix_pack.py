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

"""The packed forward and the chunked log-prob passes against the padded full-logits path.

The flash-attn varlen kernel is CUDA only, so these run the packed layout through a dense
stand-in with the kernel's semantics. What they cover is the packing plan, the unpacking and the
trainer wiring; the kernel itself is `python -m trl.trainer.prefix_pack_check`.
"""

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from trl import GRPOTrainer
from trl.trainer import prefix_pack


PROMPT, LENS, VOCAB = 11, (7, 3, 9, 1), 96


def dense_varlen(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """`flash_attn_varlen_func` as a dense masked softmax: one (query, key) segment pair at a
    time, the causal mask aligned to the bottom right when a segment has more keys than queries."""
    assert causal and dropout_p == 0.0
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    groups = q.shape[1] // k.shape[1]
    out = []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i] : cu_q[i + 1]]
        ki = k[cu_k[i] : cu_k[i + 1]].repeat_interleave(groups, dim=1)
        vi = v[cu_k[i] : cu_k[i + 1]].repeat_interleave(groups, dim=1)
        n_q, n_k = qi.shape[0], ki.shape[0]
        visible = torch.arange(n_k)[None, :] <= torch.arange(n_q)[:, None] + (n_k - n_q)
        scores = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        out.append(torch.einsum("hqk,khd->qhd", scores.masked_fill(~visible, -torch.inf).softmax(-1), vi))
    return torch.cat(out)


class Trainer:
    """The attributes the two forward seams read, around the real methods."""

    _get_last_hidden_state = GRPOTrainer._get_last_hidden_state
    _get_per_token_logps_and_entropies = GRPOTrainer._get_per_token_logps_and_entropies
    model_kwarg_keys = ()
    _is_vlm = False
    _entropy_bonus_enabled = False
    temperature = 0.7
    accelerator = SimpleNamespace(unwrap_model=lambda model: model, is_main_process=True)

    def __init__(self, prefix_pack=False, use_liger_kernel=False):
        self.prefix_pack = prefix_pack
        self.use_liger_kernel = use_liger_kernel


@pytest.fixture(autouse=True)
def dense_kernel(monkeypatch):
    monkeypatch.setattr(prefix_pack, "_varlen", dense_varlen)
    monkeypatch.setattr(prefix_pack, "_stock_attn", None)


@pytest.fixture(params=["base", "lora"])
def model(request):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(config).eval()
    if request.param == "lora":
        lora = LoraConfig(r=4, target_modules="all-linear", init_lora_weights=False, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora).eval()
    return model


@pytest.fixture
def batch():
    torch.manual_seed(1)
    b, width = len(LENS), max(LENS)
    prompt = torch.randint(0, VOCAB, (1, PROMPT)).repeat(b, 1)
    completions = torch.randint(0, VOCAB, (b, width))
    completion_mask = torch.zeros(b, width, dtype=torch.long)
    for i, n in enumerate(LENS):
        completion_mask[i, :n] = 1
    input_ids = torch.cat([prompt, completions], dim=1)
    attention_mask = torch.cat([torch.ones_like(prompt), completion_mask], dim=1)
    return input_ids, attention_mask, completion_mask.bool(), width


def test_packed_hidden_states_match_padded(model, batch):
    input_ids, attention_mask, live, width = batch
    with torch.no_grad():
        padded = Trainer()._get_last_hidden_state(model, input_ids, attention_mask, width)
        packed = Trainer(prefix_pack=True)._get_last_hidden_state(model, input_ids, attention_mask, width)
    assert prefix_pack.decoder(model).config._attn_implementation == prefix_pack.ATTENTION
    torch.testing.assert_close(packed[live], padded[live], rtol=1e-4, atol=1e-5)
    assert not packed[~live].any()


def test_mixed_prompts_take_the_padded_path(model, batch):
    input_ids, attention_mask, _, width = batch
    trainer = Trainer(prefix_pack=True)
    with torch.no_grad():
        padded = Trainer()._get_last_hidden_state(model, input_ids, attention_mask, width)
        trainer._get_last_hidden_state(model, input_ids, attention_mask, width)  # leaves the packed backend set
        input_ids = input_ids.clone()
        input_ids[1, 0] = (input_ids[1, 0] + 1) % VOCAB
        mixed = trainer._get_last_hidden_state(model, input_ids, attention_mask, width)
    assert prefix_pack.decoder(model).config._attn_implementation != prefix_pack.ATTENTION
    torch.testing.assert_close(mixed[0], padded[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("packed", [False, True])
def test_no_grad_logps_match_full_logits(model, batch, packed):
    input_ids, attention_mask, live, width = batch
    with torch.no_grad():
        ref, _, _ = Trainer()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, width)
        logps, entropies, aux_loss = Trainer(
            prefix_pack=packed, use_liger_kernel=True
        )._get_per_token_logps_and_entropies(model, input_ids, attention_mask, width, batch_size=3)
    assert entropies is None and aux_loss is None
    torch.testing.assert_close(logps[live], ref[live], rtol=1e-4, atol=1e-5)
    assert not logps[~live].any()


def test_packed_logps_and_gradient_match_full_logits(model, batch):
    input_ids, attention_mask, live, width = batch
    weight = prefix_pack.decoder(model).embed_tokens.weight.requires_grad_(True)

    def run(trainer):
        logps, entropies, _ = trainer._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, width, compute_entropy=True
        )
        (grad,) = torch.autograd.grad(logps[live].sum(), weight)
        return logps.detach(), entropies.detach(), grad

    ref_logps, ref_entropies, ref_grad = run(Trainer())
    logps, entropies, grad = run(Trainer(prefix_pack=True))
    torch.testing.assert_close(logps[live], ref_logps[live], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(entropies[live], ref_entropies[live], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=1e-5)
