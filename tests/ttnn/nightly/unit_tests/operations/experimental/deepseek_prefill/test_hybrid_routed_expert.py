# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-device PCC test for HybridRoutedExpertFfn, the one-dispatch routed expert.

The same cases as test_single_routed_expert.py, driven onto the union op instead of the module:
the sweeps, dims and saturation cases are imported from there rather than copied, so the two
files cannot drift apart.

The op is the union of moe_fused_swiglu and unified_routed_expert_ffn -- both implementations
compiled into one binary per RISC-V, run as ordered passes over the same grid, with each expert
claimed by whichever half its active-token count selects. One expert per case, as in the
reference: the half that does not claim it still launches on every core and sweeps the counts, so
a single-expert case exercises both halves and the barrier between them.

Graded against TorchExpert, not against either shipping op, so the bar does not move when the
halves disagree numerically -- the merged compute binary runs with bfp8_pack_precise, which the
fused half requires and the unified op alone does not use.

The clamped SiLU-GLU cases from the reference have no counterpart here: the fused half implements
Silu, SituGlu and SwiGluOai, so a clamped activation would leave every below-threshold expert
unserved and the op rejects it outright.

CI runs the row-major cases, which is the layout production feeds the op; the tile-layout
variants are pruned per test with ci_pruning.tiled_x_input, exactly as the reference file does.
"""

import math
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SITU, TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
    _ISL_FUNCTIONAL_SWEEP,
    _K3_SATURATION_CASES,
    _K3_TOKEN_SWEEP,
    _SITU_BETA_GATE,
    _SITU_BETA_UP,
    _TORCH_ACTIVATION,
    _isl_params,
    reshard_expert_weights_nd,
)

# Which half serves an expert. Fixed rather than read from each model's own
# ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD, because the models that carry one all carry 320 and the
# baseline model (dsv3) carries none at all -- and a case with no threshold is not this op. At 320
# the exhaustive sweep puts 0/128/256 on the fused half and 512 upward on the unified one, so both
# halves are graded across the sweep.
_THRESHOLD = 320


# Cases the union op cannot serve, strict-xfailed so they turn red the day they start working.
# A case matches when every whitespace-separated token of the key appears in the param id.
_XFAIL: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _xfail_unsupported(request):
    """Apply _XFAIL to the cases whose ids match."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for key, reason in _XFAIL.items():
        if all(token in callspec.id for token in key.split()):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


def _idx_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.uint32,
    )


def run_hybrid_routed_expert(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = False,
    weights_dram_sharded: bool = False,
    activation=None,
    weight_scale: float = 0.02,
    weights_dtype=ttnn.bfloat4_b,
    pcc_threshold: float = 0.97,
    min_cap_frac: tuple[float, float] | None = None,
    threshold: int = _THRESHOLD,
):
    """One expert over an ``allocated_tokens`` region with ``active_tokens`` live rows, served by
    one half of the union op and swept over by the other.

    Signature mirrors ``run_single_routed_expert`` so the imported case matrices apply unchanged;
    see that docstring for what each knob is for. ``threshold`` is the one addition -- it decides
    which half owns the expert.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    if activation is None:
        activation = ttnn.RoutedExpertActivation.Silu
    torch_activation = _TORCH_ACTIVATION.get(activation)
    if torch_activation is None:
        raise ValueError(f"no torch reference for {activation}; supported: {list(_TORCH_ACTIVATION)}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * weight_scale,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * weight_scale,
    }

    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    torch_expert = TorchExpert(
        emb_dim,
        hidden_dim,
        weights,
        activation=torch_activation,
        situ_beta=_SITU_BETA_GATE,
        situ_linear_beta=_SITU_BETA_UP,
    )
    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)
        if min_cap_frac is not None:
            # Same guard as the reference: without it a change to weight_scale, the dims or the
            # seed would quietly drop a saturation case back into the near-linear middle of both
            # tanhs while still passing.
            if torch_activation != ACTIVATION_SITU:
                raise ValueError(f"min_cap_frac given for {activation}, which defines no cap to measure")
            gate_out = torch.nn.functional.linear(torch_active, weights["gate_proj"])
            up_out = torch.nn.functional.linear(torch_active, weights["up_proj"])
            gate_min, up_min = min_cap_frac
            gate_frac = (gate_out.abs() > _SITU_BETA_GATE).float().mean().item()
            up_frac = (up_out.abs() > _SITU_BETA_UP).float().mean().item()
            logger.info(
                f"SiTU-GLU cap coverage: |gate|>{_SITU_BETA_GATE}: {gate_frac:.1%}, "
                f"|up|>{_SITU_BETA_UP}: {up_frac:.1%}"
            )
            assert gate_frac >= gate_min, f"gate cap coverage {gate_frac:.1%} below {gate_min:.1%}"
            assert up_frac >= up_min, f"up cap coverage {up_frac:.1%} below {up_min:.1%}"

    idx_tt = _idx_tensor(device, [0])
    counts_tt = _idx_tensor(device, [active_tokens])
    offsets_tt = _idx_tensor(device, [0])

    # TtRoutedExpert is the weight holder only; the op is called directly, because the module's
    # forward is where the two-op fallback lives and this file is about the union op.
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        activation=activation,
    )
    if weights_dram_sharded:
        reshard_expert_weights_nd(tt_expert, device)

    # ROW_MAJOR is bf16 and tilized inside the op; TILE is consumed directly as bf8. Pair the
    # dtype with the layout so each variation drives its real device path.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
    )

    tt_output = ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
        tt_input,
        offsets_tt,
        counts_tt,
        idx_tt,
        tt_expert.gate_projs,
        tt_expert.up_projs,
        tt_expert.down_projs,
        max_dispatched_tokens_per_expert=allocated_tokens,
        hybrid_token_threshold=threshold,
        compute_kernel_config=tt_expert.compute_kernel_config,
        activation=activation,
    )
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    tt_output_active = tt_output_torch[:active_tokens]

    half = "fused" if active_tokens <= threshold else "unified"
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows, {half} half): {pcc:.6f}")

    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold} ({half} half)"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", _isl_params(_ISL_FUNCTIONAL_SWEEP))
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_functional(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_hybrid_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    _isl_params(_ISL_EXHAUSTIVE_SWEEP, only_models=_ISL_EXHAUSTIVE_MODELS),
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_hybrid_routed_expert_isl_sweep(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
    weights_dram_sharded: bool,
):
    run_hybrid_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
        weights_dram_sharded=weights_dram_sharded,
    )


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("num_tokens", _K3_TOKEN_SWEEP, ids=[f"t{t}" for t in _K3_TOKEN_SWEEP])
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU routed expert is Blackhole-only")
def test_hybrid_routed_expert_k3_sweep(device, num_tokens: int, x_row_major: bool):
    """Fully-packed buffer at each token count, as in the reference's K3 sweep."""
    run_hybrid_routed_expert(
        device,
        num_tokens,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.SituGlu,
        # This sweep runs allocated == active, so the threshold has to scale with the case: the op
        # rejects one at or above max_dispatched_tokens_per_expert, where no expert could ever
        # reach the unified half. Half the count keeps both halves reachable at every size.
        threshold=max(1, num_tokens // 2),
    )


@pytest.mark.parametrize("weight_scale, weights_dtype, pcc_threshold, min_cap_frac", _K3_SATURATION_CASES)
@pytest.mark.skipif(not is_blackhole(), reason="SiTU-GLU routed expert is Blackhole-only")
def test_hybrid_routed_expert_k3_saturated(
    device,
    weight_scale: float,
    weights_dtype,
    pcc_threshold: float,
    min_cap_frac,
):
    """SiTU-GLU driven into its caps, on the half the count selects."""
    run_hybrid_routed_expert(
        device,
        _ISL_ALLOCATED_TOKENS,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        active_tokens=_ISL_ALLOCATED_TOKENS,
        x_row_major=True,
        activation=ttnn.RoutedExpertActivation.SituGlu,
        weight_scale=weight_scale,
        weights_dtype=weights_dtype,
        pcc_threshold=pcc_threshold,
        min_cap_frac=min_cap_frac,
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_threshold_zero(device):
    """threshold=0 gives the fused half an empty band, so the factory returns the unified
    descriptor and never merges. That is a different program from every other case here: no CB
    overlay, no shared semaphore block, no pass barrier -- and nothing else covers it."""
    run_hybrid_routed_expert(
        device,
        _ISL_ALLOCATED_TOKENS,
        DeepSeekV3Config.EMB_SIZE,
        DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        active_tokens=512,
        x_row_major=True,
        threshold=0,
    )


# Counts straddling _THRESHOLD, so the two halves each own experts inside ONE dispatch and the
# per-expert region offsets are all non-zero but the first.
_MULTI_EXPERT_COUNTS = [96, 512, 160, 640]

# The production dispatch buffer is NOT one per-expert maximum wide. It is the per-expert maximum
# times a capacity factor, shared by every local expert, plus one tile per expert boundary for the
# tile-aligned region starts -- init_helpers.compute_constants, fed the factor the prefill runner
# defaults PREFILL_CAPACITY_FACTOR to. Mirror that here so the op is graded on the buffer it meets
# in the model, with regions well inside a buffer several times deeper than any one of them.
_DISPATCH_BUFFER_CAPACITY_FACTOR = 8


def _dispatch_buffer_rows(max_tokens_per_expert: int, experts_per_chip: int) -> int:
    """Rows in the per-chip dispatch buffer, as compute_constants sizes it."""
    raw = max_tokens_per_expert * _DISPATCH_BUFFER_CAPACITY_FACTOR
    return raw + ttnn.TILE_SIZE * (min(raw, experts_per_chip) - 1)


def _tile_aligned_region_offsets(counts: list[int]) -> list[int]:
    """Region start per expert: the exclusive prefix sum of the tile-rounded counts, as
    offset_cumsum lays them out. Every region starts on a tile boundary whatever the counts."""
    offsets, running = [], 0
    for c in counts:
        offsets.append(running)
        running += -(-c // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return offsets


def _random_expert_counts(
    seed: int, experts_per_chip: int, max_per_expert: int, threshold: int, buffer_rows: int
) -> list[int]:
    """Random active-token count per local expert, the way a chip sees them in the model: anything
    from empty to the per-expert maximum, not tile-aligned, and fitting the dispatch buffer once
    tile-rounded. Seeded, so a failing draw reproduces.

    Log-uniform, not uniform. Expert load is heavy-tailed -- most experts see a few hundred tokens
    and a few see thousands -- and the hybrid threshold sits at a few percent of the per-expert
    maximum, so a uniform draw would hand the fused half one expert in twenty. Equal weight per
    octave spans the whole range while landing roughly two thirds of the experts at or below the
    threshold, which is the split the union was built for.

    Two nudges keep every draw a useful case. One expert is forced empty -- a zero-row region is
    the one shape the fixed-count case never has, and combine and the op both have to step over it.
    And if a draw happens to hand the unified half no expert at all, the largest count is redrawn
    above the threshold; the forced-empty expert already guarantees the fused half one.
    """
    rng = random.Random(seed)
    log_span = math.log(max_per_expert + 1)
    counts = [int(math.exp(rng.uniform(0.0, log_span))) - 1 for _ in range(experts_per_chip)]
    counts[rng.randrange(experts_per_chip)] = 0
    # Fit the tile-rounded sum into the buffer, leaving a tile per expert for the rounding itself.
    # Scale rather than clip, so the shape of the draw survives.
    fit = buffer_rows - ttnn.TILE_SIZE * experts_per_chip
    aligned = sum(-(-c // ttnn.TILE_SIZE) * ttnn.TILE_SIZE for c in counts)
    if aligned > fit:
        counts = [c * fit // aligned for c in counts]
    if not any(c > threshold for c in counts):
        counts[counts.index(max(counts))] = rng.randint(threshold + 1, max_per_expert)
    assert all(0 <= c <= max_per_expert for c in counts)
    offsets = _tile_aligned_region_offsets(counts)
    assert offsets[-1] + counts[-1] <= buffer_rows
    return counts


def _run_multi_expert(
    device,
    *,
    emb_dim: int,
    hidden_dim: int,
    activation,
    threshold: int,
    x_row_major: bool,
    passes: list[tuple[int, list[int]]],
):
    """Several experts with DISTINCT weights inside ONE dispatch, graded per expert region against
    TorchExpert. One pass per (seed, counts) entry, each on fresh weights and a fresh buffer: the
    per-expert weight addresses are re-patched on a program-cache hit, and a wrong slot there only
    shows from the second call on.

    The buffer is sized and laid out as dispatch does it in the model: capacity-factor times the
    per-expert maximum plus the alignment reserve, each region at a tile-aligned offset. The
    per-expert maximum handed to the op stays _ISL_ALLOCATED_TOKENS, as in the single-expert cases.
    Only the rows inside each region are graded; the op may write whole tiles, so the padding rows
    between regions are its own.
    """
    experts_per_chip = len(passes[0][1])
    assert all(len(counts) == experts_per_chip for _, counts in passes)
    buffer_rows = _dispatch_buffer_rows(_ISL_ALLOCATED_TOKENS, experts_per_chip)
    torch_activation = _TORCH_ACTIVATION[activation]

    for seed, counts in passes:
        offsets = _tile_aligned_region_offsets(counts)
        assert all(0 <= c <= _ISL_ALLOCATED_TOKENS for c in counts), "no expert may exceed the per-expert maximum"
        assert offsets[-1] + counts[-1] <= buffer_rows
        assert any(c <= threshold for c in counts) and any(
            c > threshold for c in counts
        ), "the point of this case is that both halves run; counts must straddle the threshold"
        logger.info(
            f"multi-expert pass seed={seed}: {experts_per_chip} experts, {sum(counts)} live rows in a "
            f"{buffer_rows}-row buffer, counts={counts}"
        )

        torch.manual_seed(seed)
        weights = [
            {
                "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
            }
            for _ in counts
        ]
        torch_input = torch.zeros(buffer_rows, emb_dim, dtype=torch.float32)
        for off, cnt in zip(offsets, counts):
            torch_input[off : off + cnt] = torch.randn(cnt, emb_dim, dtype=torch.float32)

        idx_tt = _idx_tensor(device, list(range(experts_per_chip)))
        tt_expert = TtRoutedExpert(
            mesh_device=device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=idx_tt,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=_ISL_ALLOCATED_TOKENS,
            torch_weights=weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat4_b,
            activation=activation,
        )
        tt_input = ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
        )
        tt_output = ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            tt_input,
            _idx_tensor(device, offsets),
            _idx_tensor(device, counts),
            idx_tt,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            max_dispatched_tokens_per_expert=_ISL_ALLOCATED_TOKENS,
            hybrid_token_threshold=threshold,
            compute_kernel_config=tt_expert.compute_kernel_config,
            activation=activation,
        )
        got = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))

        for e, (off, cnt) in enumerate(zip(offsets, counts)):
            half = "fused" if cnt <= threshold else "unified"
            if cnt == 0:
                logger.debug(f"expert {e} (0 rows, {half} half): empty region, nothing to grade")
                continue
            rows = torch_input[off : off + cnt]
            with torch.no_grad():
                want = TorchExpert(
                    emb_dim,
                    hidden_dim,
                    weights[e],
                    activation=torch_activation,
                    situ_beta=_SITU_BETA_GATE,
                    situ_linear_beta=_SITU_BETA_UP,
                )(rows)
            _, pcc = comp_pcc(want, got[off : off + cnt])
            logger.debug(f"expert {e} ({cnt} rows, {half} half): PCC {pcc:.6f}")
            assert pcc >= 0.97, f"expert {e} ({half} half, rows {off}..{off + cnt}) PCC {pcc:.6f}"
            assert not torch.isnan(got[off : off + cnt]).any(), f"expert {e}: NaN in output"


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_multi_expert(device, x_row_major: bool):
    """Four experts at fixed, tile-aligned counts straddling the threshold, on DeepSeek V3's shape.

    Every single-expert case leaves one thing unchecked: that each half writes the right rows for
    the right expert. A swapped region offset, a stale per-expert weight address or cross-expert
    CB state would all still pass them. This is the small, fixed-shape version of that check; the
    model-shaped random version is test_hybrid_routed_expert_model_multi_expert.
    """
    _run_multi_expert(
        device,
        emb_dim=DeepSeekV3Config.EMB_SIZE,
        hidden_dim=DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        activation=ttnn.RoutedExpertActivation.Silu,
        threshold=_THRESHOLD,
        x_row_major=x_row_major,
        # Same counts both passes: different weights and buffers are what grade the cache-hit re-patch.
        passes=[(42, _MULTI_EXPERT_COUNTS), (7, _MULTI_EXPERT_COUNTS)],
    )


# The 8x4 Galaxy every one of these models ships on: NUM_ROUTED_EXPERTS over 32 chips is the
# expert count a chip's routed-expert op really carries.
_GALAXY_CHIPS = 32

# (model, config, K axis, hidden, activation, hybrid threshold). K3's routed experts run at the
# LatentMoE width, not EMB_SIZE, and K3 does not ship the hybrid split today -- its config keeps the
# measured crossover under _MEASURED -- so this grades the op on K3's shape at that crossover, not
# whether the model dispatches it.
_MODEL_MULTI_EXPERT_CASES = [
    pytest.param(
        "glm_5_2",
        GLM52Config,
        GLM52Config.EMB_SIZE,
        GLM52Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
        GLM52Config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD,
        id="glm_5_2",
    ),
    pytest.param(
        "kimi_k2_7",
        KimiK27Config,
        KimiK27Config.EMB_SIZE,
        KimiK27Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
        KimiK27Config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD,
        id="kimi_k2_7",
    ),
    pytest.param(
        "kimi_k3",
        KimiK3Config,
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.SituGlu,
        KimiK3Config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED,
        id="kimi_k3",
    ),
]


@pytest.mark.uncollect_if(pred=ci_pruning.tiled_x_input)
@pytest.mark.parametrize("model_name, config, emb_dim, hidden_dim, activation, threshold", _MODEL_MULTI_EXPERT_CASES)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_model_multi_expert(
    device, model_name: str, config, emb_dim: int, hidden_dim: int, activation, threshold: int, x_row_major: bool
):
    """A chip's full complement of experts at each model's real shape, with random counts.

    Each pass draws a fresh random count per expert -- empty to the per-expert maximum, not
    tile-aligned, one expert always empty -- so the regions land at offsets the fixed-count case
    never produces, the buffer holds tens of thousands of rows the way the model's does, and both
    halves own several experts inside one dispatch. Counts are logged per pass; a failure names the
    seed that reproduces it.
    """
    experts_per_chip = config.NUM_ROUTED_EXPERTS // _GALAXY_CHIPS
    buffer_rows = _dispatch_buffer_rows(_ISL_ALLOCATED_TOKENS, experts_per_chip)
    passes = [
        (seed, _random_expert_counts(seed, experts_per_chip, _ISL_ALLOCATED_TOKENS, threshold, buffer_rows))
        for seed in (42, 7)
    ]
    _run_multi_expert(
        device,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        activation=activation,
        threshold=threshold,
        x_row_major=x_row_major,
        passes=passes,
    )


# Every model that measured a crossover runs 256 routed experts over 8 chips.
_MODEL_EXPERTS_PER_CHIP = 32


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
def test_hybrid_routed_expert_config_fits_default_ring(device):
    """The union program's kernel config has to fit the ring a device gets by default.

    Sized at a model's expert count, not the one expert every case above uses, because the two are
    not the same program: the reader carries three per-expert weight addresses in its runtime args
    and the writer a fourth, so the config grows about 2 KB between one expert and a real
    32-expert layer, against a ring that does not move. It clears by a few hundred bytes, so any
    growth in either half's kernel text breaks it -- and without this that break lands in a
    32-device model test rather than here.

    One weight tensor is shared by all 32 experts on purpose: this grades the program's size, not
    its output, and 32 distinct copies of the real shape cost gigabytes of host memory.
    """
    emb_dim, hidden_dim = DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE
    torch.manual_seed(42)

    shared = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    counts = [32] * 24 + [512] * 8  # straddles the threshold, so neither half is optimised away
    offsets, running = [], 0
    for c in counts:
        offsets.append(running)
        running += c
    assert running <= _ISL_ALLOCATED_TOKENS

    x = ttnn.from_torch(
        torch.randn(_ISL_ALLOCATED_TOKENS, emb_dim, dtype=torch.float32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    idx_tt = _idx_tensor(device, list(range(_MODEL_EXPERTS_PER_CHIP)))
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=_MODEL_EXPERTS_PER_CHIP,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ISL_ALLOCATED_TOKENS,
        torch_weights=[shared] * _MODEL_EXPERTS_PER_CHIP,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    try:
        ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            x,
            _idx_tensor(device, offsets),
            _idx_tensor(device, counts),
            idx_tt,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            max_dispatched_tokens_per_expert=_ISL_ALLOCATED_TOKENS,
            hybrid_token_threshold=_THRESHOLD,
            compute_kernel_config=tt_expert.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
        )
    except RuntimeError as exc:
        if "kernel config buffer" not in str(exc):
            raise
        pytest.fail(
            "the union program no longer fits the default kernel-config ring, so the op is broken "
            "on any device opened at the default worker_l1_size. Recover the bytes in kernel text "
            f"-- see the rules in hybrid_llk_shims.hpp for what may be out-of-lined.\n{exc}"
        )
