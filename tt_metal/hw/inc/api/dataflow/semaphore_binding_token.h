// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @brief Physical path a semaphore's accesses take. The host picks it automatically while
 *        building the program, from where the semaphore's binder kernels run; there is no user
 *        intervention needed. The kernel gets the pick inside its binding token. The pick is
 *        the fastest path that keeps the semaphore's operations atomic.
 *        Quasar (tt-2xx) only. Gen1 (Wormhole, Blackhole) always resolves to LOCAL_NONATOMIC.
 *
 *  - LOCAL_NONATOMIC: Stored in L1 and accessed by read-modify-write. Picked only when at most
 *                     one binder instance exists.
 *  - DM_LOCAL_CACHED: Stored in a dedicated L1 pool and accessed through the DM cache via
 *                     RISC-V AMO. Picked only when all binders are DMs on the same node where
 *                     the semaphore exists. The pool is separate so a cached AMO's cache-line
 *                     write-back cannot clobber NoC-written data: see MEM_SEM_CACHED_POOL_BASE
 *                     in dev_mem_map.h.
 *  - EXTERNAL:        Stored in L1 and accessed through atomic operations via the NOC. Picked
 *                     whenever the semaphore is reachable beyond a single node.
 *
 * @note Never access a bound semaphore's word directly (get_semaphore(), the noc_semaphore_*
 *       free functions, raw pointers), always go through the Semaphore class. A raw access is
 *       invisible to the host's mechanism choice and can silently race it.
 */
enum class SemScope : uint8_t {
    LOCAL_NONATOMIC = 0,
    DM_LOCAL_CACHED = 1,
    EXTERNAL = 2,
};

/**
 * @brief Per-binding token for a semaphore, emitted into the generated kernel header as a
 *        constexpr object: `constexpr SemaphoreBindingToken name{id, scope};`.
 *
 * Carries everything the host resolved for this binding: the semaphore id and the mechanism
 * its accesses must use. Construct a Semaphore from it, `Semaphore s(sem::name);`. The token
 * is a compile-time constant, so the mechanism folds away once the constructor is inlined.
 */
struct SemaphoreBindingToken {
    std::uint32_t id;
    SemScope scope;
};

namespace sem_internal {

/**
 * @brief One DM_LOCAL_CACHED binding on this node, as the generated header lists them for the
 *        firmware's pool entry/exit (sem_internal::init_dm_local_cached in noc_semaphore.h):
 *        the semaphore id and how many binder harts on this node take part.
 */
struct CachedSemaphore {
    std::uint32_t id;
    std::uint32_t binder_harts;
};

}  // namespace sem_internal
