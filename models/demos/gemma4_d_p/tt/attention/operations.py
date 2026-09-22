# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared attention operations for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (not the llama variant).
No Meta-format weight conversion needed. No transformation matrices needed.

Handles:
- Per-head RMSNorm (q_norm, k_norm, v_norm) via reshape trick
- Partial RoPE for global layers (split, rotate, concat)
- K=V tying (fused Q+K+K weight, standard nlp_create_qkv_heads split)
- No bias on any projection
- scaling=1.0 (no 1/sqrt(d_k))
"""

import os

from loguru import logger

import ttnn

from .weights import AttentionWeights

_ACT_WITNESS_DONE = set()


def prefill_short_lived_memcfg() -> ttnn.MemoryConfig:
    """Some ops improve overall perf by leaving their activations in L1. This function returns L1 interleaved config, unless overriden to DRAM."""
    dram_only = os.environ.get("GEMMA4_ACTIVATIONS_DRAM_ONLY", "0").lower() in ("1", "true", "yes")
    # Witness, once per process: this placement is the single biggest e2e lever here and
    # its flag was inverted by a rebase (the old GEMMA4_PREFILL_L1_ACT is dead on this
    # model). A measurement harness must be able to prove which side it actually ran.
    if dram_only not in _ACT_WITNESS_DONE:
        _ACT_WITNESS_DONE.add(dram_only)
        logger.info(
            "[DIAG] prefill activations: DRAM-only (GEMMA4_ACTIVATIONS_DRAM_ONLY=1)"
            if dram_only
            else "[DIAG] prefill activations: L1 interleaved (default)"
        )
    return ttnn.DRAM_MEMORY_CONFIG if dram_only else ttnn.L1_MEMORY_CONFIG


_ATTN_PC_LOGGED = set()


def attn_mm_pc(act, weight):
    """DIAG GEMMA4_ATTN_MM_PC: explicit program config for an attention projection.

    Exp 7. Extends to the attention projections what Exp 6 established for the MLP ones:
    a `core_grid` alone is only half the fix; `in0_block_w` plus explicit per-core
    blocking is the other half. Exp 2 gave these four matmuls a grid and it helped only
    at chunk 8192 -- the same "grid alone is partial" signature Exp 6 then explained.

    Evidence this should work (P6 capture, chunk 2048, sliding layers):
        MLP _x5376x5376 after Exp 6   122-140 us   105-121 TFLOPs
        attention projections today   121-181 us    46-75  TFLOPs   <- 1.6-2.6x behind
    Same op class, same widths, same grid. 18.06 ms of floor, ~20% of the in-layer total.

    in0_block_w must divide k_tiles, and k_tiles differs per shape here (168 / 64 / 128),
    so it is chosen per shape rather than fixed at the MLP's 14.
    """
    if os.environ.get("GEMMA4_ATTN_MM_PC", "0").lower() not in ("1", "true", "yes"):
        return None
    try:
        g = act.device().compute_with_storage_grid_size()
        gx, gy = g.x, g.y
        m_tiles = act.shape[-2] // 32
        k_tiles = act.shape[-1] // 32
        n_tiles = weight.shape[-1] // 32
    except Exception:
        return None
    if min(m_tiles, k_tiles, n_tiles) < 1:
        return None
    cap = int(os.environ.get("GEMMA4_ATTN_MM_PC_MAXBW") or 16)
    bw = max(d for d in range(1, min(k_tiles, cap) + 1) if k_tiles % d == 0)
    per_core_M = -(-m_tiles // gy)
    per_core_N = -(-n_tiles // gx)
    sh = 4
    while sh > 1 and per_core_M % sh:
        sh -= 1
    sw = 2
    while sw > 1 and per_core_N % sw:
        sw -= 1
    while sh * sw > 8 and sh > 1:  # subblock tile budget
        sh -= 1
    key = (m_tiles, k_tiles, n_tiles)
    if key not in _ATTN_PC_LOGGED:
        _ATTN_PC_LOGGED.add(key)
        logger.info(
            f"[DIAG] ATTN explicit mm cfg ENGAGED: {m_tiles}x{k_tiles}x{n_tiles} tiles "
            f"grid={gx}x{gy} in0_block_w={bw} per_core_M={per_core_M} per_core_N={per_core_N} "
            f"subblock={sh}x{sw}"
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None, kv_tied: bool = False):
    """Project to QKV, or QK when kv_tied selects the narrow tied weight."""
    w_tensor = weights.wqk if kv_tied else weights.wqkv
    _pc = attn_mm_pc(hidden_states, w_tensor)  # DIAG GEMMA4_ATTN_MM_PC (Exp 7)
    if _pc is not None:
        return ttnn.linear(hidden_states, w_tensor, memory_config=memory_config, program_config=_pc)
    return ttnn.linear(hidden_states, w_tensor, memory_config=memory_config)


def split_qkv_heads_prefill(
    xqkv_fused,
    config,
    is_global: bool,
    tp: int = 1,
    kv_replicated: bool = False,
    kv_tied: bool = False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """Split the local projection into Q, K and V head tensors.

    With kv_tied, K and V read the same projection columns but return separate tensors.
    memory_config selects storage for the resulting activations."""
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
        kv_tied=kv_tied,
    )


def apply_per_head_norm(tensor, eps, weight=None, memory_config=None):
    """Normalize each token and head independently along head_dim."""
    orig_shape = tensor.shape
    _, num_heads, seq_len, head_dim = orig_shape
    flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_len, head_dim))

    # Use HiFi4 and fp32 acc for greater accuracy
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tensor.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    normed = ttnn.rms_norm(
        flat,
        weight=weight,
        epsilon=eps,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    return ttnn.reshape(normed, orig_shape)
