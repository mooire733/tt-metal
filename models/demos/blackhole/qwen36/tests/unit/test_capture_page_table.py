# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A trace capture must not write over the KV history the sequence is still using.

``Qwen36Model.capture_verify_trace`` is not a dry run. It executes three REAL forwards -- a warm-up,
the narrow-head probe, and the captured body itself -- each over ``bucket`` zero tokens at
``chunk_start=0``, and each ending in a ``paged_fill_cache``. Handed the LIVE page table, those
writes land on the pages holding the bucket that just completed, zeroing the KV history that every
later step attends back over.

That is what made re-capturing at the anchor (``DFLASH_RECAPTURE_ON_ANCHOR``) only a partial repair:
it fixed the dead trace but introduced this, and the residual showed up as a TRANSIENT acceptance
dip that healed over ~13 steps as fresh KV refilled the new bucket (per-step accepted lengths
1,2,1,2,1,2,1,3,2,2,1,2,1 then 4,6,8,13 -- against the eager arm's 9,12,4,3,2 from the same step).

It is invisible to a traced-vs-eager comparison, which is why ``_dual_check`` reported
``argmax_agree=1.0000`` and clean taps throughout: both sides recompute from the SAME zeroed pages
and agree with each other while both are wrong. Agreement is not correctness when the two share
their corrupted input.

These assertions are on the arithmetic only -- no mesh, no model -- so they run in CI.
"""

from __future__ import annotations

import torch

from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget

BLOCK = 64
BUCKET = 128


def _table(n_blocks):
    return torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)


def test_capture_pages_never_overlap_the_live_span():
    """The pages a capture writes must be disjoint from every page the sequence can still reach."""
    per_bucket = BUCKET // BLOCK
    pt = _table(64)
    for anchor in range(0, 3000, BUCKET):
        got = TtTarget.capture_page_table(pt, anchor, BLOCK, BUCKET)
        written = set(got[0, :per_bucket].tolist())
        # Everything the sequence has already written, plus the bucket it is about to.
        live = set(pt[0, : -(-(anchor + BUCKET) // BLOCK)].tolist())
        assert not (written & live), (
            f"anchor {anchor}: capture would write pages {sorted(written & live)} that the "
            f"sequence is using -- this is the KV-zeroing bug"
        )


def test_only_the_buckets_own_entries_are_redirected():
    """``paged_fill_cache`` maps the fill onto the FIRST ``bucket // block_size`` entries, so the
    rest of the table must survive untouched -- the replay re-stages it and reads those entries."""
    per_bucket = BUCKET // BLOCK
    pt = _table(64)
    got = TtTarget.capture_page_table(pt, BUCKET, BLOCK, BUCKET)
    assert torch.equal(got[:, per_bucket:], pt[:, per_bucket:]), "entries past the bucket were altered"
    assert got.shape == pt.shape


def test_degrades_to_the_live_table_rather_than_aliasing():
    """With no spare pages left the table comes back UNCHANGED.

    Aliasing a page the sequence is about to use would trade this bug for a worse one, so the
    degraded case must be "no protection", never "wrong pages".
    """
    pt = _table(64)
    got = TtTarget.capture_page_table(pt, 64 * BLOCK - BUCKET, BLOCK, BUCKET)
    assert torch.equal(got, pt), "should have returned the live table untouched when out of spare pages"


def test_does_not_mutate_the_caller_s_table():
    """The target holds one page table for the whole generation; the capture must not edit it."""
    pt = _table(64)
    before = pt.clone()
    TtTarget.capture_page_table(pt, BUCKET, BLOCK, BUCKET)
    assert torch.equal(pt, before), "capture_page_table mutated the page table in place"
