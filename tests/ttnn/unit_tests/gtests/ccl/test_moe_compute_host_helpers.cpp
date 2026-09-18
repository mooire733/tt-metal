// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <exception>

#include "gtest/gtest.h"
#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"

namespace {

using tt::tt_fabric::Topology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;
using ttnn::experimental::prim::detail::compute_fused_source_buffer_layout;
using ttnn::operations::ccl::common::get_neighbors;

// A host-only view of `shape`: every device slot is remote, which is all get_neighbors needs
// (it reads the shape and walks coordinates).
MeshDeviceView make_view(const MeshShape& shape) {
    std::vector<tt::tt_metal::distributed::MaybeRemote<tt::tt_metal::IDevice*>> devices(
        shape.mesh_size(), tt::tt_metal::distributed::MaybeRemote<tt::tt_metal::IDevice*>::remote());
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
    fabric_node_ids.reserve(shape.mesh_size());
    for (uint32_t chip = 0; chip < shape.mesh_size(); ++chip) {
        fabric_node_ids.emplace_back(tt::tt_fabric::MeshId{0}, chip);
    }
    return MeshDeviceView(shape, devices, fabric_node_ids);
}

constexpr uint32_t kBf16Bytes = 2;

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutMatchesPhysicalShard) {
    // Producer shard already split to one data-parallel column: [2 buffers x 32 rows, 640] in BF16
    // for hidden 2560 over four combine columns. One shard row is one token segment, so the ring
    // entry is half the shard height.
    constexpr uint32_t hidden_size = 2560;
    constexpr uint32_t data_parallel_cores = 4;
    constexpr uint32_t token_segment_width = hidden_size / data_parallel_cores;  // 640
    constexpr uint32_t source_shard_height = 64;
    constexpr uint32_t source_shard_width = token_segment_width;
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t token_segment_size_bytes = token_segment_width * kBf16Bytes;                       // 1280
    constexpr uint32_t source_buffer_size_bytes = source_shard_height * source_shard_width * kBf16Bytes;  // 81920

    const auto layout = compute_fused_source_buffer_layout(
        source_shard_height,
        source_shard_width,
        token_segment_width,
        source_buffer_size_bytes,
        token_segment_size_bytes,
        num_buffers);

    EXPECT_EQ(layout.rows_per_buffer, 32u);
    EXPECT_EQ(layout.buffer_block_size_bytes, 40960u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 81920u);
    EXPECT_LE(layout.circular_buffer_size_bytes, source_buffer_size_bytes);

    // Producer and consumer toggle between offsets 0 and buffer_block_size_bytes; the last token
    // segment of either block stays inside the circular buffer.
    EXPECT_EQ(layout.buffer_block_size_bytes, layout.rows_per_buffer * token_segment_size_bytes);
    EXPECT_EQ(layout.circular_buffer_size_bytes, num_buffers * layout.buffer_block_size_bytes);
    EXPECT_LE(
        layout.buffer_block_size_bytes + layout.rows_per_buffer * token_segment_size_bytes,
        layout.circular_buffer_size_bytes);
}

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutFullWidthShardCountsSegmentRows) {
    // moe_compute's own tilize-output shard, [2 buffers x 32 rows, hidden 7168] in BF16 (the deepseek
    // single-card nightly shape), consumed by four 1792-element combine columns. Each shard row holds
    // four token segments, so a ring entry is 32 x 4 = 128 token-segment rows and buffer 1 starts
    // half-way through the shard (458752 B), where the host readback expects it.
    constexpr uint32_t hidden_size = 7168;
    constexpr uint32_t data_parallel_cores = 4;
    constexpr uint32_t token_segment_width = hidden_size / data_parallel_cores;  // 1792
    constexpr uint32_t source_shard_height = 64;
    constexpr uint32_t source_shard_width = hidden_size;
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t token_segment_size_bytes = token_segment_width * kBf16Bytes;                       // 3584
    constexpr uint32_t source_buffer_size_bytes = source_shard_height * source_shard_width * kBf16Bytes;  // 917504

    const auto layout = compute_fused_source_buffer_layout(
        source_shard_height,
        source_shard_width,
        token_segment_width,
        source_buffer_size_bytes,
        token_segment_size_bytes,
        num_buffers);

    EXPECT_EQ(layout.rows_per_buffer, 128u);
    EXPECT_EQ(layout.buffer_block_size_bytes, 458752u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 917504u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, source_buffer_size_bytes);
    EXPECT_EQ(layout.buffer_block_size_bytes, source_buffer_size_bytes / num_buffers);
    EXPECT_EQ(layout.buffer_block_size_bytes, layout.rows_per_buffer * token_segment_size_bytes);
}

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutSingleBufferUsesWholeShard) {
    const auto layout = compute_fused_source_buffer_layout(
        /*source_shard_height=*/32,
        /*source_shard_width=*/512,
        /*token_segment_width=*/512,
        /*source_buffer_size_bytes=*/32 * 1024,
        /*token_segment_size_bytes=*/1024,
        /*num_buffers=*/1);
    EXPECT_EQ(layout.rows_per_buffer, 32u);
    EXPECT_EQ(layout.buffer_block_size_bytes, 32u * 1024u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 32u * 1024u);
}

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutRejectsBadInputs) {
    // Shard height not divisible by the buffer count.
    EXPECT_THROW(compute_fused_source_buffer_layout(33, 512, 512, 33 * 1024, 1024, 2), std::exception);
    // Shard width not divisible by the token segment width.
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 1000, 640, 64 * 1000 * 2, 1280, 2), std::exception);
    // Token segment size not a whole number of bytes per element over its width.
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 640, 640, 64 * 640 * 2, 1281, 2), std::exception);
    // Circular buffer larger than the L1 bank that backs it.
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 64 * 1024 - 1, 1024, 2), std::exception);
    // Zero arguments.
    EXPECT_THROW(compute_fused_source_buffer_layout(0, 512, 512, 1024, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 0, 512, 64 * 1024, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 0, 64 * 1024, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 0, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 64 * 1024, 0, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 64 * 1024, 1024, 0), std::exception);
}

TEST(MoEComputeHostHelpers, GetNeighborsAxisOfExtentOneIsEmpty) {
    // An axis of extent 1 is a valid trivial topology: no neighbours, no directions, no error,
    // under Linear and under Ring (where the only wrap-around candidate would be the coordinate
    // itself). This is what makes moe_compute's combine degenerate to a local write on a 1xN
    // expert-parallel mesh with cluster_axis=0.
    const std::array<bool, 4> no_directions = {false, false, false, false};
    const auto row_view = make_view(MeshShape(1, 4));
    for (uint32_t col = 0; col < 4; ++col) {
        for (const auto topology : {Topology::Linear, Topology::Ring}) {
            const auto [neighbors, directions] = get_neighbors(row_view, MeshCoordinate(0, col), topology, 0);
            EXPECT_TRUE(neighbors.empty()) << "col " << col;
            EXPECT_EQ(directions, no_directions) << "col " << col;
        }
    }
    const auto column_view = make_view(MeshShape(4, 1));
    for (uint32_t row = 0; row < 4; ++row) {
        const auto [neighbors, directions] = get_neighbors(column_view, MeshCoordinate(row, 0), Topology::Linear, 1);
        EXPECT_TRUE(neighbors.empty()) << "row " << row;
        EXPECT_EQ(directions, no_directions) << "row " << row;
    }
    const auto single_view = make_view(MeshShape(1, 1));
    for (const auto axis : {0u, 1u}) {
        const auto [neighbors, directions] = get_neighbors(single_view, MeshCoordinate(0, 0), Topology::Linear, axis);
        EXPECT_TRUE(neighbors.empty()) << "axis " << axis;
        EXPECT_EQ(directions, no_directions) << "axis " << axis;
    }
}

TEST(MoEComputeHostHelpers, GetNeighborsAxisWithExtentKeepsNeighbours) {
    // The multi-device axis of the same views is unchanged: a line end has one neighbour, an
    // interior coordinate two, a ring end two through the wrap. directions = {E, W, N, S}.
    const auto row_view = make_view(MeshShape(1, 4));
    {
        const auto [neighbors, directions] = get_neighbors(row_view, MeshCoordinate(0, 0), Topology::Linear, 1);
        ASSERT_EQ(neighbors.size(), 1u);
        EXPECT_EQ(neighbors[0], MeshCoordinate(0, 1));
        EXPECT_EQ(directions, (std::array<bool, 4>{true, false, false, false}));
    }
    {
        const auto [neighbors, directions] = get_neighbors(row_view, MeshCoordinate(0, 1), Topology::Linear, 1);
        ASSERT_EQ(neighbors.size(), 2u);
        EXPECT_EQ(neighbors[0], MeshCoordinate(0, 2));
        EXPECT_EQ(neighbors[1], MeshCoordinate(0, 0));
        EXPECT_EQ(directions, (std::array<bool, 4>{true, true, false, false}));
    }
    {
        const auto [neighbors, directions] = get_neighbors(row_view, MeshCoordinate(0, 0), Topology::Ring, 1);
        ASSERT_EQ(neighbors.size(), 2u);
        EXPECT_EQ(neighbors[0], MeshCoordinate(0, 1));
        EXPECT_EQ(neighbors[1], MeshCoordinate(0, 3));
        EXPECT_EQ(directions, (std::array<bool, 4>{true, true, false, false}));
    }
    const auto column_view = make_view(MeshShape(4, 1));
    {
        const auto [neighbors, directions] = get_neighbors(column_view, MeshCoordinate(3, 0), Topology::Linear, 0);
        ASSERT_EQ(neighbors.size(), 1u);
        EXPECT_EQ(neighbors[0], MeshCoordinate(2, 0));
        EXPECT_EQ(directions, (std::array<bool, 4>{false, false, true, false}));
    }
    // Both axes requested on a 1x1 mesh is still an error: nothing to talk to on any axis.
    EXPECT_THROW(
        get_neighbors(make_view(MeshShape(1, 1)), MeshCoordinate(0, 0), Topology::Linear, std::nullopt),
        std::exception);
}

}  // namespace
