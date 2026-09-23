// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include "../../../../common/llk_sfpu/ckernel_sfpu_erf_bw_bf16_blackhole.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_erf_bw_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::ErfBwBf16Config, ITERATIONS>();
}
inline void init_erf_bw_tt_poly_bf16() { ckernel::sfpu::ttpoly::init_config_tile<ttpoly_generated::ErfBwBf16Config>(); }
template <int ITERATIONS = 32>
inline void calculate_erf_bw_gradient_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_config_tile<ttpoly_generated::ErfBwBf16Config::Gradient, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
