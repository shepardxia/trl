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

"""Shared-prefix forward for GRPO: encode a group's prompt once instead of once per completion.

Every completion in a microbatch comes from the same prompt, so the padded `[B, P+C]` layout
recomputes P identical prompt tokens B times. This packs the microbatch into one sequence -- the
prompt, then every completion back to back -- and gives each completion the prompt plus itself
through one flash-attn varlen call, so the body sees each prompt token once. The hidden states are
unpacked into `[B, C, H]`, leaving the loss call unchanged.

Requires group-contiguous microbatches, so the trainer skips its generation-batch shuffle under
`prefix_pack`; that cannot change the gradient while `steps_per_generation ==
gradient_accumulation_steps` and `num_iterations == 1`. A microbatch whose rows do not share one
prompt takes the padded path.

`python -m trl.trainer.prefix_pack_check` checks the varlen call against the dense mask it
implements, then the packed hidden states and their gradient against the padded ones.
"""

from typing import NamedTuple

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..losses.fused_linear_ppo import _SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE
from .utils import _ChunkedLogProbFunction


ATTENTION = "prefix_flash"  # attention implementation name this module registers
# transformers dispatches this name to the Hub kernel; no locally built flash-attn needed.
FLASH = "kernels-community/flash-attn2"


def decoder(model):
    """Walk DDP / peft / CausalLM wrappers down to the module that owns the decoder layers."""
    m = model
    for _ in range(8):
        if hasattr(m, "layers"):
            return m
        m = getattr(m, "module", None) or getattr(m, "base_model", None) or getattr(m, "model")
    raise AttributeError("no decoder found under the model wrappers")


def lm_head(model):
    m = model
    for _ in range(8):
        h = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
        if h is not None:
            return h
        m = getattr(m, "module", None) or getattr(m, "model")
    raise AttributeError("no lm_head found under the model wrappers")


# The varlen plan of the microbatch being forwarded, set by `packed_hidden` and read by the
# attention interface below, which the decoder calls once per layer with no way to pass it.
_ctx = {}

# The backend the model was loaded with, captured before the first packed forward overwrites
# it, so a microbatch that cannot be packed falls back to what the config asked for.
_stock_attn = None

_varlen = None


def _varlen_func():
    """flash-attn's varlen entry point, from the Hub kernel transformers itself dispatches to.

    Dao-AILab publishes no flash-attn 2 wheel past torch 2.9, and none is needed: transformers
    maps `flash_attention_2` to `kernels-community/flash-attn2`, whose `flash_attn_varlen_func`
    takes the same arguments. Fetched once, on the first attention call.
    """
    global _varlen
    if _varlen is None:
        from kernels import get_kernel

        _varlen = get_kernel(FLASH, version=3).flash_attn_varlen_func
    return _varlen


def prefix_flash_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    softcap=None,
    **kwargs,
):
    """Attention over the packed sequence: each completion sees the prompt plus itself.

    The prompt is stored once, so its keys and values are gathered per completion into a
    varlen batch whose query segments are shorter than their key segments; flash-attn aligns
    the causal mask to the bottom right in that case, which is exactly "whole prompt visible,
    causal within the completion". Only keys and values are gathered -- every projection and
    MLP still runs on each prompt token once, which is the point of packing.
    """
    out = _varlen_func()(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1).index_select(0, _ctx["kv"]),
        value[0].transpose(0, 1).index_select(0, _ctx["kv"]),
        _ctx["cu_q"],
        _ctx["cu_k"],
        _ctx["max_q"],
        _ctx["max_k"],
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
        window_size=(-1, -1) if sliding_window is None else (sliding_window - 1, 0),
        softcap=0.0 if softcap is None else softcap,
    )
    return out.unsqueeze(0), None


# Registering only the attention function and not a mask function is what keeps transformers
# from materialising a [1, 1, T, T] causal mask for the packed length.
ALL_ATTENTION_FUNCTIONS[ATTENTION] = prefix_flash_attention


class Packed(NamedTuple):
    """One microbatch as a single sequence, with what inverts it and what attends to it."""

    ids: torch.Tensor  # [1, P + N] prompt tokens then every completion, back to back
    pos: torch.Tensor  # [1, P + N] position ids, each completion restarting at P
    row: torch.Tensor  # [N] padded-layout row of each completion token
    local: torch.Tensor  # [N] padded-layout position of each completion token
    src: torch.Tensor  # [N] packed position of the state that predicts it
    plan: dict  # varlen segmentation, read by `prefix_flash_attention`
    b: int
    p: int


def pack(prompt_ids, prompt_mask, completion_ids, completion_mask):
    """Pack the microbatch into one sequence. Every index is built once here, not per layer."""
    device = prompt_ids.device
    p_row = prompt_ids[0][prompt_mask[0].bool()]
    p = int(p_row.numel())
    lens = completion_mask.sum(1)
    b, n = lens.numel(), int(lens.sum())
    starts = lens.cumsum(0) - lens

    row = torch.repeat_interleave(torch.arange(b, device=device), lens)
    local = torch.arange(n, device=device) - torch.repeat_interleave(starts, lens)
    # A completion's first token is predicted by the last prompt token, which is packed once;
    # every later token is predicted by its own predecessor in the packed sequence.
    src = torch.where(local > 0, torch.arange(p - 1, p - 1 + n, device=device), p - 1)

    # Keys run the prompt for its own query segment, then prompt ++ completion for each.
    seg = lens + p
    within = torch.arange(int(seg.sum()), device=device) - torch.repeat_interleave(seg.cumsum(0) - seg, seg)
    plan = dict(
        kv=torch.cat(
            [
                torch.arange(p, device=device),
                torch.where(within < p, within, torch.repeat_interleave(starts, seg) + within),
            ]
        ),
        cu_q=torch.cat([lens.new_zeros(1), lens.new_full((1,), p), p + lens.cumsum(0)]).int(),
        cu_k=torch.cat([lens.new_zeros(1), lens.new_full((1,), p), p + seg.cumsum(0)]).int(),
        max_q=max(p, int(lens.max())),
        max_k=p + int(lens.max()),
    )

    return Packed(
        ids=torch.cat([p_row, completion_ids[completion_mask.bool()]]).unsqueeze(0),
        pos=torch.cat([torch.arange(p, device=device), local + p]).unsqueeze(0),
        row=row,
        local=local,
        src=src,
        plan=plan,
        b=b,
        p=p,
    )


def unpack(hidden, pk, width):
    """Scatter the state predicting each completion token into the padded `[B, width, H]`."""
    out = hidden.new_zeros(pk.b, width, hidden.shape[-1])
    out[pk.row, pk.local] = hidden[0, pk.src]
    mask = torch.zeros(pk.b, width, dtype=torch.long, device=hidden.device)
    mask[pk.row, pk.local] = 1
    targets = torch.zeros_like(mask)
    targets[pk.row, pk.local] = pk.ids[0, pk.p :]
    return out, mask, targets


def packed_hidden(model, prompt_ids, prompt_mask, completion_ids, completion_mask, width):
    """Body forward over the packed microbatch.

    Returns the hidden state that predicts each completion token, its loss mask and the ids
    it predicts, all in the `[B, width, *]` layout the loss expects.
    """
    global _stock_attn
    dec = decoder(model)
    if _stock_attn is None:
        _stock_attn = dec.config._attn_implementation
    pk = pack(prompt_ids, prompt_mask, completion_ids, completion_mask)
    _ctx.clear()
    _ctx.update(pk.plan)
    dec.config._attn_implementation = ATTENTION
    h = dec(input_ids=pk.ids, attention_mask=None, position_ids=pk.pos, use_cache=False).last_hidden_state
    return unpack(h, pk, width)


def restore_attention(model):
    """Hand the model back the backend it was loaded with, ahead of a padded forward.

    The packed backend is set per call and never restored after one, because gradient
    checkpointing recomputes layers during the backward, where the stock backend would be
    handed a packed batch.
    """
    dec = decoder(model)
    dec.config._attn_implementation = _stock_attn or dec.config._attn_implementation


def chunked_head(head, config, hidden, targets, mask, temperature):
    """Log-probs and entropies for the completion tokens without materialising `[N, vocab]` logits.

    `_ChunkedLogProbFunction` streams the lm_head over vocab chunks with an online logsumexp
    and carries its own backward, so this is one differentiable op rather than a loop -- which
    also means every DDP rank enqueues the same single gradient regardless of how wide its
    completions are. The vocab chunk is the fused loss's own, so the two heads agree.
    """
    b, width, h = hidden.shape
    keep = mask.bool().reshape(-1)
    lp_flat, ent_flat = _ChunkedLogProbFunction.apply(
        hidden.reshape(-1, h)[keep].contiguous(),
        head.weight,
        getattr(head, "bias", None),
        targets.reshape(-1)[keep].contiguous(),
        temperature,
        _SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE,
        getattr(config, "final_logit_softcapping", None),
        getattr(config, "logit_scale", 1.0) or 1.0,
    )

    def _scatter(flat):
        out = flat.new_zeros(b * width)
        out[keep] = flat
        return out.view(b, width)

    return _scatter(lp_flat), _scatter(ent_flat)


def shares_one_prompt(prompt_ids, prompt_mask):
    """Whether every row carries the same prompt, in one device->host sync."""
    lens = prompt_mask.sum(1)
    same = (lens == lens[0]).all() & (prompt_ids == prompt_ids[0:1]).logical_or(prompt_mask == 0).all()
    return bool(same)
