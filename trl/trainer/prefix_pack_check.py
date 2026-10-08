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

"""Correctness checks and a forward/backward benchmark for the shared-prefix forward.

    python -m trl.trainer.prefix_pack_check                      # varlen call, then the model body
    python -m trl.trainer.prefix_pack_check --attention_grad     # kernel backward
    python -m trl.trainer.prefix_pack_check --grad batch.json --adapter <checkpoint>
    python -m trl.trainer.prefix_pack_check --bench

Compare norms against the sdpa control, never against an absolute: the floor moves with the input.
"""

import torch

from .prefix_pack import (
    FLASH,
    _ctx,
    chunked_head,
    decoder,
    lm_head,
    pack,
    packed_hidden,
    prefix_flash_attention,
)


def _rel(a, c):
    """Relative error as a norm ratio: a single outlier element must not set the verdict."""
    a, c = a.float(), c.float()
    return float(((a - c).norm() / c.norm()).detach())


def _packed_mask(t, p, lens, device):
    """[t, t] boolean: the prompt attends causally to itself, and each completion attends to
    the whole prompt plus causally to itself."""
    idx = torch.arange(t, device=device)
    mask = torch.zeros(t, t, dtype=torch.bool, device=device)
    mask[:p, :p] = idx[:p, None] >= idx[None, :p]
    off = p
    for n in lens:
        mask[off:off + n, :p] = True
        mask[off:off + n, off:off + n] = idx[:n, None] >= idx[None, :n]
        off += n
    return mask


def _dense_attention(q, k, v, p, lens, scaling):
    """The packed attention written out as a dense masked softmax, in float32: the
    specification the varlen call has to meet."""
    t = q.shape[0]
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1).float()
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1).float()
    mask = _packed_mask(t, p, lens, q.device)
    scores = torch.einsum("thd,shd->hts", q.float(), k) * scaling
    return torch.einsum("hts,shd->thd", scores.masked_fill(~mask, -torch.inf).softmax(-1), v)


def check_attention_grad(p=900, lens=(1300, 1700, 2100, 1100), heads=32, kv_heads=8, dim=128,
                         dtype=torch.bfloat16):
    """The varlen call's backward against the dense float32 specification, with torch's sdpa
    on the same mask as a second kernel.

    Random projections at production lengths; the upstream gradient is random too, so every
    query, key and value receives one. Each kernel's dq, dk and dv are scored as a norm ratio
    against the float32 gradient of `_dense_attention`.
    """
    import torch.nn.functional as F

    dev = "cuda"
    b, width = len(lens), max(lens)
    mask = torch.zeros(b, width, dtype=torch.long, device=dev)
    for i, n in enumerate(lens):
        mask[i, :n] = 1
    pk = pack(torch.zeros(b, p, dtype=torch.long, device=dev),
              torch.ones(b, p, dtype=torch.long, device=dev),
              torch.zeros(b, width, dtype=torch.long, device=dev), mask)
    _ctx.clear()
    _ctx.update(pk.plan)

    t, scaling = p + sum(lens), dim ** -0.5
    q = torch.randn(1, heads, t, dim, dtype=dtype, device=dev)
    k = torch.randn(1, kv_heads, t, dim, dtype=dtype, device=dev)
    v = torch.randn(1, kv_heads, t, dim, dtype=dtype, device=dev)
    up = torch.randn(t, heads, dim, dtype=torch.float32, device=dev)

    def grads(fn, *xs):
        xs = [x.detach().clone().requires_grad_(True) for x in xs]
        (fn(*xs).float() * up).sum().backward()
        return [x.grad.float() for x in xs]

    ref = grads(lambda q, k, v: _dense_attention(q[0].transpose(0, 1), k[0].transpose(0, 1),
                                                 v[0].transpose(0, 1), p, lens, scaling),
                q.float(), k.float(), v.float())
    varlen = grads(lambda q, k, v: prefix_flash_attention(None, q, k, v, None,
                                                           scaling=scaling)[0][0], q, k, v)
    dense_mask = _packed_mask(t, p, lens, dev)
    sdpa = grads(lambda q, k, v: F.scaled_dot_product_attention(
        q, k, v, attn_mask=dense_mask, scale=scaling, enable_gqa=True)[0].transpose(0, 1),
        q, k, v)
    print(f"attention backward vs dense float32 ({dtype}), prompt {p}, completions {list(lens)}")
    for name, g in (("varlen", varlen), ("sdpa", sdpa)):
        print(f"  {name:8s}" + "  ".join(f"{n} {_rel(a, r):.2e}"
                                         for n, a, r in zip(("dq", "dk", "dv"), g, ref)))


def check_attention(p=211, lens=(97, 34, 120, 61, 15, 88), heads=32, kv_heads=8, dim=128,
                    dtype=torch.float16):
    """Check the varlen call against the dense mask it is supposed to implement.

    Runs on random projections rather than a model, so nothing amplifies a geometry fault
    and nothing hides one: the only error left is the kernel's own accumulation, which at
    float16 is around 1e-3. Also exercises `pack`, since the plan under test is its output.
    """
    dev = "cuda"
    b, width = len(lens), max(lens)
    mask = torch.zeros(b, width, dtype=torch.long, device=dev)
    for i, n in enumerate(lens):
        mask[i, :n] = 1
    pk = pack(torch.zeros(b, p, dtype=torch.long, device=dev),
              torch.ones(b, p, dtype=torch.long, device=dev),
              torch.zeros(b, width, dtype=torch.long, device=dev), mask)
    _ctx.clear()
    _ctx.update(pk.plan)

    t = p + sum(lens)
    q = torch.randn(1, heads, t, dim, dtype=dtype, device=dev)
    k = torch.randn(1, kv_heads, t, dim, dtype=dtype, device=dev)
    v = torch.randn(1, kv_heads, t, dim, dtype=dtype, device=dev)
    out = prefix_flash_attention(None, q, k, v, None, scaling=dim ** -0.5)[0][0]
    ref = _dense_attention(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1),
                           p, lens, dim ** -0.5)
    print(f"attention vs dense mask ({dtype}): {_rel(out, ref):.2e}")


def check_model(model_id="Qwen/Qwen3-8B", p=211, lens=(97, 34, 120, 61, 15, 88),
                dtype="bfloat16"):
    """Check the packed body and its gradient against the padded ones on real weights.

    Same prompt, same completions, same weights and the same chunked head on both sides, so
    the only thing under test is the packing. The padded sdpa forward is the control: it is
    the same maths as the padded flash forward, so what it differs by is this dtype's kernel
    noise through 36 layers, and the packed path has to sit at that floor. Tokens come from
    this file, because the model is off-distribution on random ids.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), attn_implementation=FLASH,
        device_map={"": 0}).eval().requires_grad_(False)
    dec, head = decoder(model), lm_head(model)
    dec.embed_tokens.weight.requires_grad_(True)

    b, width, dev = len(lens), max(lens), "cuda"
    need = p + b * width
    text = AutoTokenizer.from_pretrained(model_id)(__doc__, return_tensors="pt").input_ids[0]
    stream = text.repeat(need // text.numel() + 1)[:need].to(dev)
    prompt_ids = stream[:p].unsqueeze(0).repeat(b, 1)
    prompt_mask = torch.ones(b, p, dtype=torch.long, device=dev)
    completion_ids = stream[p:].view(b, width)
    completion_mask = torch.zeros(b, width, dtype=torch.long, device=dev)
    for i, n in enumerate(lens):
        completion_mask[i, :n] = 1
    targets = completion_ids * completion_mask

    def padded(impl):
        dec.config._attn_implementation = impl
        return dec(input_ids=torch.cat([prompt_ids, completion_ids], 1),
                   attention_mask=torch.cat([prompt_mask, completion_mask], 1),
                   use_cache=False).last_hidden_state[:, p - 1:p + width - 1]

    def grad_of(hidden):
        logps, _ = chunked_head(head, dec.config, hidden, targets, completion_mask, 1.0)
        return torch.autograd.grad((logps * completion_mask).sum(),
                                   dec.embed_tokens.weight)[0]

    packed = packed_hidden(model, prompt_ids, prompt_mask, completion_ids,
                            completion_mask, width)[0]
    g_packed = grad_of(packed)
    ref, control = padded(FLASH), padded("sdpa")
    g_ref, g_control = grad_of(ref), grad_of(control)

    live = completion_mask.bool()
    print(f"{model_id} {dtype}, prompt {p}, completions {list(lens)}")
    print(f"{'':12s}{'packed':>10s}{'sdpa (floor)':>14s}")
    print(f"{'hidden':12s}{_rel(packed[live], ref[live]):>10.2e}"
          f"{_rel(control[live], ref[live]):>14.2e}")
    print(f"{'embed grad':12s}{_rel(g_packed, g_ref):>10.2e}{_rel(g_control, g_ref):>14.2e}")
    print("per row     " + " ".join(f"{_rel(packed[i, :n], ref[i, :n]):.1e}"
                                    for i, n in enumerate(lens)))


def check_grad(batch_path, model_id="Qwen/Qwen3-8B", adapter=None, max_completion_length=4096):
    """The LoRA gradient of the production GRPO loss on real groups, packed against TRL's own path.

    `batch_path` is a JSON list of groups, each `{"prompt": [messages], "completions": [text],
    "advantages": [float]}`. Each group runs as one microbatch under the training setup: the
    adapter, liger's module patches, every layer checkpointed, dropout off so the arms see one
    function, and TRL's fused DAPO loss normalised by the group's token count. The reference is
    TRL's stock `_get_last_hidden_state` on the flash kernel. Two floors: the same arm run again
    (the backward's nondeterministic accumulation) and the stock path on sdpa (kernel noise). A
    group's gradient is a sum over completions whose advantages sum to zero, so it cancels and
    its relative error runs far above the per-completion one. Each arm also reports the mean NLL of the
    sampled tokens and the mean entropy at their positions: for completions sampled at
    temperature 1 by the weights under test the two agree, so a gap is a sampler mismatch.
    """
    import json
    import types

    from liger_kernel.transformers import _apply_liger_kernel_to_instance
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..losses import FusedLinearGRPOLoss
    from .grpo_trainer import GRPOTrainer

    def trainer(prefix_pack):
        return types.SimpleNamespace(model_kwarg_keys=(), _is_vlm=False, prefix_pack=prefix_pack)

    packed, stock = trainer(True), trainer(False)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=FLASH, device_map={"": 0})
    _apply_liger_kernel_to_instance(model=model)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.train()
    dec, head = decoder(model), lm_head(model)
    set_checkpointing(model, dec, "all")
    params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = FusedLinearGRPOLoss(
        beta=0.0, epsilon_low=0.2, epsilon_high=0.28, temperature=1.0, use_ref_model=False,
        loss_type="dapo", max_completion_length=max_completion_length, compiled=False)

    dev = "cuda"
    for g, group in enumerate(json.load(open(batch_path))):
        text = tok.apply_chat_template(group["prompt"], add_generation_prompt=True, tokenize=False)
        p_ids = tok(text, add_special_tokens=False).input_ids
        comps = [(tok(c, add_special_tokens=False).input_ids + [tok.eos_token_id])[
            :max_completion_length] for c in group["completions"]]
        b, width = len(comps), max(map(len, comps))
        prompt_ids = torch.tensor([p_ids] * b, device=dev)
        prompt_mask = torch.ones_like(prompt_ids)
        completion_ids = torch.full((b, width), tok.pad_token_id, device=dev)
        completion_mask = torch.zeros(b, width, dtype=torch.long, device=dev)
        for i, c in enumerate(comps):
            completion_ids[i, :len(c)] = torch.tensor(c, device=dev)
            completion_mask[i, :len(c)] = 1
        advantages = torch.tensor(group["advantages"], dtype=torch.float32, device=dev)
        input_ids = torch.cat([prompt_ids, completion_ids], 1)
        attention_mask = torch.cat([prompt_mask, completion_mask], 1)

        def arm(trainer, impl):
            dec.config._attn_implementation = impl
            h = GRPOTrainer._get_last_hidden_state(trainer, model, input_ids, attention_mask, width)
            loss, _ = loss_fn(_input=h, lin_weight=head.weight, selected_token_ids=completion_ids,
                              attention_mask=completion_mask, advantages=advantages,
                              bias=getattr(head, "bias", None),
                              num_items_in_batch=completion_mask.sum())
            with torch.no_grad():
                logps, ents = chunked_head(head, dec.config, h.detach(),
                                            completion_ids * completion_mask, completion_mask, 1.0)
            loss.backward()
            grad = torch.cat([p.grad.double().flatten().cpu() for p in params])
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            m = completion_mask.float()
            row_lp = ((logps * m).sum(1) / m.sum(1)).cpu()
            nll, ent = float(-(logps * m).sum() / m.sum()), float((ents * m).sum() / m.sum())
            return float(loss), grad, row_lp, nll, ent

        _ctx.clear()
        out = {"packed": arm(packed, FLASH)}
        assert _ctx, "the packed arm fell back to the stock path"
        out.update(stock=arm(stock, FLASH), repeat=arm(stock, FLASH), sdpa=arm(stock, "sdpa"))
        ref = out["stock"]
        print(f"group {g}: prompt {len(p_ids)}, {b} completions {min(map(len, comps))}-{width} "
              f"(sum {sum(map(len, comps))}), advantages {advantages.min():.2f}..{advantages.max():.2f}")
        print(f"  {'':8s}{'loss':>12s}{'grad rel err':>14s}{'grad cosine':>13s}"
              f"{'grad norm':>11s}{'max row logp diff':>19s}{'token nll':>11s}{'entropy':>9s}")
        for name, (loss, grad, row_lp, nll, ent) in out.items():
            cos = float(grad @ ref[1] / (grad.norm() * ref[1].norm()))
            print(f"  {name:8s}{loss:>12.6f}{_rel(grad, ref[1]):>14.2e}{cos:>13.6f}"
                  f"{float(grad.norm()):>11.4e}{float((row_lp - ref[2]).abs().max()):>19.2e}"
                  f"{nll:>11.4f}{ent:>9.4f}", flush=True)


def set_checkpointing(model, dec, mode):
    """Gradient checkpointing at a granularity between "every layer" and "off".

    `all` is what the trainer does: every decoder layer recomputes its whole forward in the
    backward. `skipK` leaves one layer in K unwrapped, so it keeps its activations and skips
    its recompute. `sacK` keeps every layer wrapped but saves one matmul output in K and
    recomputes the rest, which is the finer-grained version of the same trade. Both buy time
    with memory, linearly in the fraction kept.
    """
    if mode == "none":
        model.gradient_checkpointing_disable()
        return
    kwargs = {"use_reentrant": False}
    if mode.startswith("sac"):
        from torch.utils.checkpoint import (CheckpointPolicy,
                                            create_selective_checkpoint_contexts)
        every = int(mode[len("sac"):])
        matmul = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
                  torch.ops.aten.bmm.default}

        def context():
            # The policy runs twice over the same op sequence, once in the forward and once
            # in the recompute, and the two must reach the same verdict op for op. They need
            # separate counters: a shared one keeps counting into the recompute and the saved
            # tensors then line up against the wrong operators.
            seen = {True: 0, False: 0}

            def policy(ctx, op, *args, **kwargs_):
                if op not in matmul:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                seen[ctx.is_recompute] += 1
                return (CheckpointPolicy.MUST_SAVE
                        if seen[ctx.is_recompute] % every == 0
                        else CheckpointPolicy.PREFER_RECOMPUTE)

            return create_selective_checkpoint_contexts(policy)

        kwargs["context_fn"] = context
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    if mode.startswith("skip"):
        k = int(mode[len("skip"):])
        for i, layer in enumerate(dec.layers):
            layer.gradient_checkpointing = i % k != 0


def bench(model_id="Qwen/Qwen3-8B", p=1200, b=16, mean_len=2300, reps=5,
          use_kernels=False, arms=("padded:all", "packed:all"), liger=True,
          liger_loss=True):
    """Time one microbatch's forward and backward, packed against padded.

    Everything the arms do not name is shared: same weights, same LoRA, same gradient
    checkpointing, same lengths. Generation and reward are left out because they are
    identical on both sides and are most of a step's wall clock, so a training run buries
    the difference this measures. `liger` and `liger_loss` are separate axes -- one is the
    RMSNorm/SwiGLU/rope module patches, the other is the fused GRPO loss against TRL's own
    chunked log-prob function -- because "liger" names both and they need not agree.
    """
    import time

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=FLASH,
        device_map={"": 0}, use_kernels=use_kernels)
    if liger:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model)
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear"))
    model.train()
    dec, head = decoder(model), lm_head(model)

    dev = "cuda"
    lens = [max(256, min(4096, mean_len + (i - b // 2) * 150)) for i in range(b)]
    width, vocab = max(lens), int(dec.config.vocab_size)
    prompt_ids = torch.randint(0, vocab, (1, p), device=dev).repeat(b, 1)
    prompt_mask = torch.ones(b, p, dtype=torch.long, device=dev)
    completion_ids = torch.randint(0, vocab, (b, width), device=dev)
    completion_mask = torch.zeros(b, width, dtype=torch.long, device=dev)
    for i, n in enumerate(lens):
        completion_mask[i, :n] = 1
    advantages = torch.randn(b, device=dev)
    if liger_loss:
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
        fused = LigerFusedLinearGRPOLoss(
            beta=0.0, epsilon_low=0.2, epsilon_high=0.28, temperature=1.0,
            use_ref_model=False, loss_type="dapo", max_completion_length=width)

    def loss_of(hidden, mask, targets):
        if liger_loss:
            return fused(_input=hidden, lin_weight=head.weight, selected_token_ids=targets,
                         attention_mask=mask, advantages=advantages,
                         bias=getattr(head, "bias", None))[0]
        # TRL's non-liger branch: chunked log-probs, then the DAPO objective elementwise.
        logps, _ = chunked_head(head, dec.config, hidden, targets, mask, 1.0)
        ratio = torch.exp(logps - logps.detach())
        per_token = -torch.min(ratio * advantages[:, None],
                               ratio.clamp(0.8, 1.28) * advantages[:, None])
        return (per_token * mask).sum() / mask.sum()

    def padded():
        dec.config._attn_implementation = FLASH
        h = dec(input_ids=torch.cat([prompt_ids, completion_ids], 1),
                attention_mask=torch.cat([prompt_mask, completion_mask], 1),
                use_cache=False).last_hidden_state[:, p - 1:p + width - 1]
        return h, completion_mask, completion_ids * completion_mask

    def packed():
        return packed_hidden(model, prompt_ids, prompt_mask, completion_ids,
                              completion_mask, width)

    def one(build):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden, mask, targets = build()
        loss = loss_of(hidden, mask, targets)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        return t1 - t0, time.perf_counter() - t1

    print(f"{model_id}: prompt {p}, {b} completions {min(lens)}-{max(lens)} "
          f"(sum {sum(lens)}), use_kernels={use_kernels}, liger={liger}, "
          f"liger_loss={liger_loss}, {reps} reps")
    for arm in arms:
        name, mode = arm.split(":")
        build = {"padded": padded, "packed": packed}[name]
        set_checkpointing(model, dec, mode)
        try:
            one(build)
            torch.cuda.reset_peak_memory_stats()
            t = sorted(one(build) for _ in range(reps))[reps // 2]
            print(f"  {arm:16s} fwd {t[0]:6.2f}s  bwd {t[1]:6.2f}s  total {sum(t):6.2f}s  "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:5.1f}GB", flush=True)
        except torch.OutOfMemoryError as e:
            print(f"  {arm:16s} out of memory: {str(e).split(chr(10))[0]}", flush=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Correctness checks for the packed forward.")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--prompt_len", type=int, default=211)
    ap.add_argument("--lens", type=int, nargs="+", default=[97, 34, 120, 61, 15, 88])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attention_only", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="time a production-shaped microbatch instead of checking it")
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--mean_len", type=int, default=2300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--use_kernels", action="store_true")
    ap.add_argument("--no_liger", action="store_true",
                    help="skip liger's RMSNorm/SwiGLU/rope module patches")
    ap.add_argument("--no_liger_loss", action="store_true",
                    help="use TRL's chunked log-prob head instead of liger's fused loss")
    ap.add_argument("--arms", nargs="+", default=["padded:all", "packed:all"],
                    help="<padded|packed>:<all|none|everyK|sac> per arm")
    ap.add_argument("--grad", metavar="BATCH_JSON",
                    help="compare the LoRA gradient of the GRPO loss on real groups, packed "
                         "against TRL's stock path (see check_grad)")
    ap.add_argument("--adapter", help="LoRA checkpoint for --grad")
    ap.add_argument("--attention_grad", action="store_true",
                    help="check the varlen and sdpa backward against the dense specification")
    a = ap.parse_args()
    if a.attention_grad:
        check_attention_grad()
    elif a.grad:
        check_grad(a.grad, a.model, a.adapter)
    elif a.bench:
        bench(a.model, a.prompt_len, a.microbatch, a.mean_len, a.reps, a.use_kernels,
              tuple(a.arms), not a.no_liger, not a.no_liger_loss)
    else:
        check_attention(a.prompt_len, tuple(a.lens))
        if not a.attention_only:
            check_model(a.model, a.prompt_len, tuple(a.lens), a.dtype)
