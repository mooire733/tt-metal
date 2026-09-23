// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include "../../../../common/llk_sfpu/ckernel_sfpu_hardswish_bw_bf16_wormhole_b0.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_hardswish_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::HardswishBwBf16Config, ITERATIONS>();
}
inline void init_hardswish_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_config_tile<ttpoly_generated::HardswishBwBf16Config>();
}
template <int ITERATIONS = 32>
inline void calculate_hardswish_bw_gradient_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::HardswishBwBf16Config::Gradient, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
