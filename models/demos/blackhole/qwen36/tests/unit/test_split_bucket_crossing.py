# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can the anchor crossing be TRACED instead of run eagerly? If so, the recapture goes away.

THE PROBLEM. ``TtTarget.forward`` crosses an anchor by running the completed bucket as one
``length == ANCHOR`` forward, and ``_run`` sends that down the EAGER path (``length < self.ANCHOR``
is false). That eager forward allocates device buffers while the verify trace is parked, and lands
on the trace's own output -- which ``capture_verify_trace`` allocates INSIDE the capture window
(``out = _body()`` between begin/end). Every traced verify after a crossing then returns all-zero
logits. Re-capturing the trace at each anchor repairs it, but costs a measured 1.86-2.29 s per
crossing -- a tax every 128 tokens, ~10 % of a 200-token generation and worse as length grows.

The codebase's own rule for traces, from tt_transformers/tt/generator.py:

    "Allocation-free outside the capture window by construction -- everything it binds to was
     allocated by _prepare_trace_prefill before any trace existed."

DFlash breaks that rule in exactly one place: the crossing. Make the crossing allocation-free and
no recapture is needed.

THE PROPOSAL this file tests. A whole bucket does not have to be one forward. Split it into two
64-row halves and each half is ``< ANCHOR``, so the EXISTING verify trace serves it: ``chunk_start``
is staged per replay (``stage_verify_inputs``), and ``paged_fill_cache`` requires only 64-row block
alignment, not ANCHOR alignment. GDN already decomposes its scan into 32-row chunks internally, so
splitting at 64 should be arithmetically equivalent -- "should be" is what this file checks.

WHAT IS COMPARED. Not the GDN state tensors directly, but the thing that actually matters: run the
NEXT block after the crossing and compare its logits and taps. Logits decide the tokens, taps decide
what the drafter sees, and the whole acceptance cliff lives in those two. If the split crossing
leaves state equivalent to the eager one, they match.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_split_bucket_crossing.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
ANCHOR = 128
HALF = 64
BLOCK = 16
TAP_LAYERS = [0, 5, 10, 20, 30, 40]


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _to_host(taps):
    return [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=-1)).float() for t in taps]


@pytest.mark.xfail(
    reason="REFUTED 2026-09-18: two traced 64-row halves are NOT equivalent to one 128-row "
    "bucket -- argmax agreement 0.7500, taps decaying with depth (L0=0.9998 L5=0.9984 "
    "L10=0.9969 L20=0.9912 L30=0.9836 L40=0.9844). Kept as the record of a dead hypothesis. "
    "Confound worth noting before anyone retries: each half runs in an ANCHOR-sized bucket, "
    "so it writes 64 rows of PADDING into the KV pages beside its 64 real rows. That is an "
    "artifact of reusing the 128-row trace for a 64-row job; a dedicated bucket-64 trace "
    "would not have it -- but that is a second trace, which is the other option anyway.",
    strict=False,
)
@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_split_crossing_matches_whole_bucket(mesh_device, device_params, reset_seeds, ensure_gc):
    """Two traced 64-row halves must leave the same state as one eager 128-row bucket."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    model.set_residual_taps(TAP_LAYERS, keep_on_device=True)

    g = torch.Generator().manual_seed(11)
    tokens = torch.randint(1000, 2000, (1, ANCHOR + BLOCK), generator=g, dtype=torch.int32)
    tail = tokens[:, ANCHOR : ANCHOR + BLOCK]

    def _tail_eager():
        """The block AFTER the crossing, run eagerly from whatever state is current."""
        lg = model.prefill_block_all_logits(tail, page_table, actual_len=BLOCK, chunk_start=ANCHOR, bucket=ANCHOR)
        return lg, _to_host(model.take_taps(BLOCK))

    # ---- REFERENCE: the crossing as it is done today, one eager whole bucket.
    # Run the whole thing twice: everything must be COMPILED before the capture below, because a
    # program first compiled with a trace parked hangs the process instead of raising.
    for _ in range(2):
        model._reset_gdn_state_for_new_sequence()
        model.prefill_block_all_logits(tokens[:, :ANCHOR], page_table, actual_len=ANCHOR, chunk_start=0, bucket=ANCHOR)
        model.take_taps(ANCHOR)
        ref_logits, ref_taps = _tail_eager()

    # Also compile the 64-row eager halves, so the same shapes exist before the capture and the
    # eager-vs-traced question is not confounded by a first compile.
    for lo in (0, HALF):
        model.prefill_block_all_logits(
            tokens[:, lo : lo + HALF], page_table, actual_len=HALF, chunk_start=lo, bucket=ANCHOR
        )
        model.take_taps(HALF)

    # ---- capture the verify trace, exactly as the shipping path does.
    model.capture_verify_trace(page_table, ANCHOR, capture_chunk_start=0)

    def _traced_half(lo):
        buf = torch.zeros(1, ANCHOR, dtype=torch.int32)
        buf[:, :HALF] = tokens[:, lo : lo + HALF]
        model.verify_traced(buf, HALF, lo, page_table, ANCHOR)
        model.take_taps(HALF)

    # ---- PROPOSAL: the same crossing as two TRACED halves, with no GDN restore between them so
    # the second continues the first's recurrence.
    model._reset_gdn_state_for_new_sequence()
    _traced_half(0)
    _traced_half(HALF)
    split_logits, split_taps = _tail_eager()

    lg_ok, lg_pcc = comp_pcc(ref_logits, split_logits, 0.99)
    agree = (ref_logits.argmax(-1) == split_logits.argmax(-1)).float().mean().item()
    logger.info(f"split crossing: LOGITS pcc {lg_pcc}, argmax agreement {agree:.4f}")

    tap_pccs = []
    for layer, e, t in zip(TAP_LAYERS, ref_taps, split_taps):
        ok, pcc = comp_pcc(e, t, 0.99)
        kind = "attn" if model.args.is_full_attention_layer(layer) else "gdn "
        tap_pccs.append((layer, pcc, ok, kind))
        logger.info(f"split crossing: TAP L{layer:<2} [{kind}] pcc {pcc}")

    print(f"\n>>> split-bucket crossing: logits pcc {lg_pcc}, argmax agree {agree:.4f}")
    print(">>> taps: " + ", ".join(f"L{l}[{k.strip()}]={p:.4f}" for l, p, _, k in tap_pccs) + "\n")

    # Argmax is the gate that matters: acceptance is an exact argmax match, so anything less than
    # full agreement would show up directly as rejected drafts.
    assert agree == 1.0, f"split crossing changes the tokens: argmax agreement {agree:.4f}"
    assert lg_ok, f"split crossing diverges on logits, pcc {lg_pcc}"
    for layer, pcc, ok, _k in tap_pccs:
        assert ok, f"split crossing diverges on tap layer {layer}, pcc {pcc}"
