/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <op/scan/fixed_page_filter_mask.hpp>

#include <cudf/column/column_factories.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>

namespace sirius::op::scan {
namespace {

constexpr int kRowsPerTile = 256;

struct fixed_page_mask_tile {
  void const* data{nullptr};
  std::size_t input_offset{0};
  std::size_t output_offset{0};
  std::size_t num_rows{0};
  cudf::type_id type_id{cudf::type_id::EMPTY};
};

template <typename T>
__device__ bool eval_integral(T value, fixed_page_simple_predicate predicate)
{
  auto const v = static_cast<int64_t>(value);
  bool keep = true;
  if (predicate.has_lower) {
    keep = keep && (predicate.lower_inclusive ? v >= predicate.lower_i64 : v > predicate.lower_i64);
  }
  if (predicate.has_upper) {
    keep = keep && (predicate.upper_inclusive ? v <= predicate.upper_i64 : v < predicate.upper_i64);
  }
  return keep;
}

template <typename T>
__device__ bool eval_unsigned(T value, fixed_page_simple_predicate predicate)
{
  auto const v = static_cast<uint64_t>(value);
  bool keep = true;
  if (predicate.has_lower) {
    auto const lower = static_cast<uint64_t>(predicate.lower_i64);
    keep = keep && (predicate.lower_inclusive ? v >= lower : v > lower);
  }
  if (predicate.has_upper) {
    auto const upper = static_cast<uint64_t>(predicate.upper_i64);
    keep = keep && (predicate.upper_inclusive ? v <= upper : v < upper);
  }
  return keep;
}

template <typename T>
__device__ bool eval_floating(T value, fixed_page_simple_predicate predicate)
{
  auto const v = static_cast<double>(value);
  bool keep = true;
  if (predicate.has_lower) {
    keep = keep && (predicate.lower_inclusive ? v >= predicate.lower_f64 : v > predicate.lower_f64);
  }
  if (predicate.has_upper) {
    keep = keep && (predicate.upper_inclusive ? v <= predicate.upper_f64 : v < predicate.upper_f64);
  }
  return keep;
}

__global__ void fixed_page_simple_filter_kernel(fixed_page_mask_tile const* tiles,
                                                std::size_t num_tiles,
                                                fixed_page_simple_predicate predicate,
                                                uint8_t* out)
{
  auto const tile_idx = static_cast<std::size_t>(blockIdx.x);
  if (tile_idx >= num_tiles) { return; }
  auto const tile = tiles[tile_idx];
  auto const row = static_cast<std::size_t>(threadIdx.x);
  if (row >= tile.num_rows) { return; }

  bool keep = false;
  auto const input_idx = tile.input_offset + row;
  switch (tile.type_id) {
    case cudf::type_id::INT8:
      keep = eval_integral(static_cast<int8_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::INT16:
      keep = eval_integral(static_cast<int16_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
      keep = eval_integral(static_cast<int32_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DECIMAL64:
      keep = eval_integral(static_cast<int64_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::UINT8:
      keep = eval_unsigned(static_cast<uint8_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::UINT16:
      keep = eval_unsigned(static_cast<uint16_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::UINT32:
      keep = eval_unsigned(static_cast<uint32_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::UINT64:
      keep = eval_unsigned(static_cast<uint64_t const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::FLOAT32:
      keep = eval_floating(static_cast<float const*>(tile.data)[input_idx], predicate);
      break;
    case cudf::type_id::FLOAT64:
      keep = eval_floating(static_cast<double const*>(tile.data)[input_idx], predicate);
      break;
    default: keep = false; break;
  }
  out[tile.output_offset + row] = keep ? uint8_t{1} : uint8_t{0};
}

}  // namespace

std::unique_ptr<cudf::column> make_fixed_page_simple_filter_mask(
  std::vector<fixed_page_mask_segment> const& segments,
  fixed_page_simple_predicate predicate,
  std::size_t output_rows,
  rmm::cuda_stream_view stream)
{
  auto bool_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                static_cast<cudf::size_type>(output_rows),
                                                cudf::mask_state::UNALLOCATED,
                                                stream);
  if (output_rows == 0) { return bool_col; }

  std::vector<fixed_page_mask_tile> host_tiles;
  for (auto const& segment : segments) {
    std::size_t consumed = 0;
    while (consumed < segment.num_rows) {
      auto const rows = std::min<std::size_t>(kRowsPerTile, segment.num_rows - consumed);
      host_tiles.push_back(fixed_page_mask_tile{segment.data,
                                                segment.input_offset + consumed,
                                                segment.output_offset + consumed,
                                                rows,
                                                segment.type_id});
      consumed += rows;
    }
  }

  rmm::device_uvector<fixed_page_mask_tile> device_tiles(host_tiles.size(), stream);
  cudaMemcpyAsync(device_tiles.data(),
                  host_tiles.data(),
                  host_tiles.size() * sizeof(fixed_page_mask_tile),
                  cudaMemcpyHostToDevice,
                  stream.value());
  fixed_page_simple_filter_kernel<<<static_cast<unsigned int>(host_tiles.size()),
                                    kRowsPerTile,
                                    0,
                                    stream.value()>>>(device_tiles.data(),
                                                      host_tiles.size(),
                                                      predicate,
                                                      bool_col->mutable_view().data<uint8_t>());
  return bool_col;
}

}  // namespace sirius::op::scan
