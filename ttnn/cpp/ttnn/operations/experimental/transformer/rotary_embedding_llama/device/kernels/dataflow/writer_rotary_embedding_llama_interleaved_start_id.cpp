// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

FORCE_INLINE void zero_tile_at(uint32_t l1_write_addr, uint32_t tile_bytes) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = 0;
    }
}

void kernel_main() {
    Noc noc;

    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);

    const auto s = TensorAccessor(tensor::output);

    DataflowBuffer dfb_out(dfb::out);
    // "zero" is writer-private staging (fill once, read back as a NoC source): a Scratchpad, not a
    // self-loop DFB. Its entry size equals the output tile size, so reuse dfb_out's tile_bytes rather
    // than the (now absent) DFB's get_entry_size(). It holds Wt tiles, filled once at the base — no
    // FIFO, no wrap (reserve/push/wait/pop each spanned the whole buffer, so the indices stayed 0).
    Scratchpad<uint32_t> zero(scratch::zero);

    const uint32_t tile_bytes = dfb_out.get_entry_size();

    const uint32_t zero_base_addr = zero.get_base_address();
    uint32_t zero_l1_write_addr = zero_base_addr;
    for (uint32_t j = 0; j < Wt; j++) {
        zero_tile_at(zero_l1_write_addr, tile_bytes);
        zero_l1_write_addr += tile_bytes;
    }
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // Quasar DM: the fill above is CPU stores that land in L1D/L2; the NoC writes below source the zero
    // tiles from TL1 directly. Flush the filled region so the NoC copies see zeros. No-op on WH/BH.
    // Matches the fill_rm / pad Scratchpad conversions (#51763).
    flush_l2_cache_range(static_cast<uintptr_t>(zero_base_addr), static_cast<size_t>(Wt * tile_bytes));
#endif

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                uint32_t output_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                const bool write_rotary_output = seq_tile < rotary_Ht;
                if (write_rotary_output) {
                    dfb_out.wait_front(Wt);
                }

                // Rotary output comes from dfb_out; the padding tail is sourced from the zero staging
                // buffer at its base. Both have tile_bytes stride, so the transfer size is uniform.
                uint32_t l1_read_addr = write_rotary_output ? dfb_out.get_read_ptr() : zero_base_addr;
                for (uint32_t j = 0; j < Wt; j++) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l1_read_addr), s, tile_bytes, {}, {.page_id = output_curr_idx});
                    l1_read_addr += tile_bytes;
                    output_curr_idx++;
                }
                noc.async_write_barrier();

                if (write_rotary_output) {
                    dfb_out.pop_front(Wt);
                }
            }
        }
    }
}
