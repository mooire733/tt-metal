# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does a SMALLER masked bucket buy throughput? The biggest single lever on the verify.

A traced verify replays the WHOLE bucket whatever the real span is, because a capture bakes shapes.
So verifying a 16-token block 20 rows into a bucket still costs a full 128-row forward, and
tests/perf/test_traced_verify_host_breakdown.py measures that replay at 107.1 ms of a ~204 ms step
at 99 % device utilization -- it cannot be dispatched away, only made smaller.

Halving ANCHOR should roughly halve it, and should also shrink `stage` (fewer staged bytes) and the
narrow head's window. What it costs back: max_block() truncates any block that would cross the
boundary, so with block 16 and bucket 64 about a quarter of start positions lose rows, and
crossings arrive twice as often. Whether the replay saving beats the acceptance loss is exactly
what this file measures, and it is why ANCHOR is not simply a smaller constant in the source.

NO CROSSING IN EITHER ARM. The budget is chosen so `5 + budget` stays inside ONE bucket at the
SMALLEST anchor tested, so this isolates replay cost from the crossing machinery -- which at the
time of writing is the unsolved part (an eager whole-bucket forward under a parked trace kills the
trace; re-capturing repairs it at 1.86-2.29 s a time). A smaller bucket makes crossings more
frequent, so this measurement is necessary but not sufficient: it says what the replay saving is
worth, not what a long generation would net.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_dflash_anchor_size_ab.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 250_000_000
PRODUCTION_MS_PER_TOK = 56.0  # text_demo.py traced_128 -- 17.87 tok/s
PROMPT = "The capital of France is"
BUDGET = 50  # 5 + 50 = 55 < 64, so no crossing at either anchor
# DFLASH_BLOCK overrides the drafter's own block_size. The bucket A/B showed the verify is
# nearly row-INSENSITIVE (128 -> 64 rows saved 5 ms of a 299 ms step), i.e. ~192 ms of
# per-layer fixed overhead across 64 layers. If a step costs the same whatever it verifies,
# then committing MORE tokens per step is the cheap lever and making steps faster is the
# expensive one. Acceptance 7.5 at the current step time is already +40 % over production.
BLOCK = int(os.environ.get("DFLASH_BLOCK", "0")) or None
ANCHORS = [128, 64]
CAP = 128
REPEATS = 3


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_throughput_vs_anchor_size(mesh_device, device_params, reset_seeds, ensure_gc):
    """Same prompt and budget at each bucket size; report tok/s against production."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from transformers import AutoTokenizer

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    anchor = TtTarget.ANCHOR  # set by DFLASH_ANCHOR; one arm per PROCESS (two 27B loads in one
    # process risks OOM, and the bucket sizes the capture and the staged buffers anyway).
    assert prompt.shape[1] + BUDGET < anchor * 2, "budget must stay inside one bucket -- no crossing"

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    drafter = TtDrafter(
        TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path), tt_ccl=model.tt_ccl, ctx_capacity=CAP),
        target,
    )
    # Every width the loop can ask for must be compiled BEFORE any capture; with a block wider
    # than the drafter's own block_size the default 1..block_size range is not enough.
    drafter.drafter.warm_block_widths(widths=range(1, (BLOCK or drafter.block_size) + 1))
    # One eager generation first: enable_traced_verify's precondition, and it compiles every shape
    # the measured runs touch so nothing compiles under a parked trace.
    dflash_generate(drafter, target, prompt, max_new_tokens=BUDGET, block_size=BLOCK)

    # DFLASH_DRAFTER_TRACE=1. The drafter is 120 ms/step against 10.5 ms of device time -- ~9 %
    # utilization, host-dispatch-bound at ~180 dispatches x ~0.6 ms
    # (tests/perf/test_dflash_drafter_wall_time.py). Tracing it was measured at 0.85x, i.e. SLOWER,
    # but that was at ctx_capacity 512, where stage_step's two [1,1,q_len,C+32] masks are most of
    # the staged bytes (tests/perf/test_dflash_drafter_trace_breakdown.py names exactly that as the
    # suspect). At C=128 those masks are a quarter the size, so the verdict is worth re-taking
    # rather than inheriting.
    #
    # Captured BEFORE the verify trace and after the eager generation: every program is already
    # compiled by then, so neither capture's warm-up compiles anything with a trace parked.
    if os.environ.get("DFLASH_DRAFTER_TRACE") == "1":
        # The capture bakes q_len, so it must match the block the loop will actually use or
        # every step falls back to eager.
        drafter.enable_traced_draft(q_len=BLOCK or drafter.block_size)
    target.enable_traced_verify(narrow_head=True)

    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=BUDGET, block_size=BLOCK, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        steps = len(stats.acceptance_lengths)
        samples.append((n / dt, stats.mean_acceptance_length, dt * 1000 / max(steps, 1), steps))

    tok_s = sum(s[0] for s in samples) / len(samples)
    acc = samples[0][1]  # greedy: identical every repeat
    step_ms = sum(s[2] for s in samples) / len(samples)
    ratio = tok_s / (1000 / PRODUCTION_MS_PER_TOK)
    logger.info(
        f">>>>> anchor {anchor:3d}: {tok_s:5.2f} tok/s, acceptance {acc:5.3f} over {samples[0][3]} steps, "
        f"step {step_ms:6.1f} ms, {ratio:4.2f}x production"
    )
    print(
        f"\n>>> ANCHOR={anchor} drafter_trace={os.environ.get('DFLASH_DRAFTER_TRACE', '0')} "
        f"tok/s={tok_s:.2f} acceptance={acc:.3f} step_ms={step_ms:.1f} "
        f"vs_production={ratio:.2f}x samples={[f'{s[0]:.2f}' for s in samples]}"
    )
    print(f">>> target for +40%: {1.4 * 1000 / PRODUCTION_MS_PER_TOK:.2f} tok/s\n")
