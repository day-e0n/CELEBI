/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scan_manager/variable_width_page_index.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/search.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <iterator>

namespace sirius::scan_manager {

namespace {
/// Target number of offset samples per column-chunk. 2048 keeps the strided
/// copy at ~12us and the map at ~16KiB of host memory, while bounding the byte
/// error of a page boundary to one bucket's skew -- for a 10M-row chunk that is
/// one sample per ~4900 rows, i.e. well under 1% of a 16MiB page for any TPC-H
/// string column. Raising it buys accuracy roughly linearly in transfer time.
constexpr std::size_t kOffsetSampleBudget = 2048;
}  // namespace

variable_width_chunk_page_index index_variable_width_column_pages(
  std::vector<std::size_t> const& row_byte_lengths, std::size_t page_size_bytes)
{
  variable_width_chunk_page_index idx;
  if (row_byte_lengths.empty() || page_size_bytes == 0) { return idx; }

  std::size_t const n = row_byte_lengths.size();
  std::size_t row     = 0;
  while (row < n) {
    std::size_t const start_row = row;
    std::size_t bytes           = 0;
    std::size_t num_rows        = 0;
    while (row < n) {
      std::size_t const next_len = row_byte_lengths[row];
      // A page already holding >=1 row stops before a row that would push it
      // over budget -- that row starts the next page instead. A brand-new
      // page always takes at least one row, even an oversized one, so a
      // single giant value doesn't stall the cursor forever.
      if (num_rows > 0 && bytes + next_len > page_size_bytes) { break; }
      bytes += next_len;
      ++num_rows;
      ++row;
      if (num_rows == 1 && bytes > page_size_bytes) {
        // That lone row is already oversized; give it its own page rather
        // than silently absorbing whatever comes next too.
        break;
      }
    }
    variable_width_column_page page;
    page.page_index = idx.pages.size();
    page.start_row  = start_row;
    page.num_rows   = num_rows;
    page.num_bytes  = bytes;
    idx.page_start_rows.push_back(start_row);
    idx.pages.push_back(std::move(page));
  }
  return idx;
}

std::int64_t variable_width_row_byte_map::total_bytes() const noexcept
{
  return byte_prefix.empty() ? 0 : byte_prefix.back();
}

std::size_t variable_width_row_byte_map::sample_row(std::size_t i) const noexcept
{
  if (byte_prefix.empty() || i >= byte_prefix.size() - 1) { return num_rows; }
  // Every sample but the last sits on a stride multiple; the last is pinned to
  // num_rows so byte_prefix.back() is the chunk's exact total (see the header).
  return std::min(i * stride, num_rows);
}

std::int64_t variable_width_row_byte_map::bytes_before(std::size_t row) const noexcept
{
  if (!valid()) { return 0; }
  if (row >= num_rows) { return total_bytes(); }

  // Bucket containing `row`, clamped so `hi` always stays a valid sample index.
  auto lo = std::min(row / stride, byte_prefix.size() - 2);
  auto const hi = lo + 1;
  auto const lo_row = sample_row(lo);
  auto const hi_row = sample_row(hi);
  if (hi_row <= lo_row || row <= lo_row) { return byte_prefix[lo]; }

  // Linear interpolation within the bucket: exact at both endpoints, and off by
  // at most this bucket's internal byte skew in between.
  auto const span_rows  = static_cast<std::int64_t>(hi_row - lo_row);
  auto const span_bytes = byte_prefix[hi] - byte_prefix[lo];
  auto const into_rows  = static_cast<std::int64_t>(row - lo_row);
  return byte_prefix[lo] + (span_bytes * into_rows) / span_rows;
}

std::int64_t variable_width_row_byte_map::bytes_in_range(std::size_t begin_row,
                                                         std::size_t end_row) const noexcept
{
  if (end_row <= begin_row) { return 0; }
  return std::max<std::int64_t>(0, bytes_before(end_row) - bytes_before(begin_row));
}

std::vector<std::size_t> cut_pages_by_byte_budget(variable_width_row_byte_map const& map,
                                                  std::size_t page_size_bytes,
                                                  std::size_t per_row_overhead_bytes)
{
  std::vector<std::size_t> starts;
  if (!map.valid() || page_size_bytes == 0) { return starts; }

  auto const budget = static_cast<std::int64_t>(page_size_bytes);
  // Real footprint of rows [begin, end): chars bytes plus the offsets child's
  // fixed per-row cost. Budgeting on chars alone is what let narrow columns
  // degenerate to one page per chunk -- see kVariableWidthOffsetsOverheadPerRow.
  auto cost = [&](std::size_t begin, std::size_t end) {
    return map.bytes_in_range(begin, end) +
           static_cast<std::int64_t>(end - begin) *
             static_cast<std::int64_t>(per_row_overhead_bytes);
  };

  std::size_t start = 0;
  while (start < map.num_rows) {
    starts.push_back(start);
    // Largest `end` with cost(start, end) <= budget, by binary search over the
    // map's monotone cumulative byte function -- O(log N) per page boundary,
    // pure host arithmetic against an already-host-resident map.
    std::size_t lo = start + 1;  // a page always takes at least one row, so an
    std::size_t hi = map.num_rows;  // oversized single row can't stall the cursor
    std::size_t best = lo;
    while (lo <= hi) {
      auto const mid = lo + (hi - lo) / 2;
      if (cost(start, mid) <= budget) {
        best = mid;
        lo   = mid + 1;
      } else {
        if (mid == start + 1) { break; }
        hi = mid - 1;
      }
    }
    start = std::max(best, start + 1);
  }
  return starts;
}

std::size_t variable_width_row_byte_skiplist::stride_of(std::size_t lvl) const noexcept
{
  std::size_t stride = base_stride;
  for (std::size_t i = 0; i < lvl; ++i) { stride *= fanout; }
  return stride;
}

std::int64_t variable_width_row_byte_skiplist::total_bytes() const noexcept
{
  return (level.empty() || level.front().empty()) ? 0 : level.front().back();
}

std::size_t variable_width_row_byte_skiplist::footprint_bytes() const noexcept
{
  std::size_t total = pages.size() * sizeof(variable_width_page_boundary);
  for (auto const& lv : level) { total += lv.size() * sizeof(std::int64_t); }
  return total;
}

std::int64_t variable_width_row_byte_skiplist::bytes_before(std::size_t row) const noexcept
{
  if (!valid()) { return 0; }
  if (row >= num_rows) { return total_bytes(); }

  auto const& lv0 = level.front();
  auto const lo   = std::min(row / base_stride, lv0.size() - 2);
  auto const hi   = lo + 1;
  auto const lo_row = std::min(lo * base_stride, num_rows);
  auto const hi_row = (hi + 1 >= lv0.size()) ? num_rows : std::min(hi * base_stride, num_rows);
  if (hi_row <= lo_row || row <= lo_row) { return lv0[lo]; }
  // base_stride == 1 never reaches here: lo_row == row exactly.
  auto const span_rows  = static_cast<std::int64_t>(hi_row - lo_row);
  auto const span_bytes = lv0[hi] - lv0[lo];
  return lv0[lo] + (span_bytes * static_cast<std::int64_t>(row - lo_row)) / span_rows;
}

std::size_t variable_width_row_byte_skiplist::find_page_end(
  std::size_t start_row, std::int64_t budget, std::size_t per_row_overhead) const noexcept
{
  if (!valid() || start_row >= num_rows) { return num_rows; }
  auto const base    = bytes_before(start_row);
  auto const per_row = static_cast<std::int64_t>(per_row_overhead);
  auto cost = [&](std::size_t r) {
    return (bytes_before(r) - base) + static_cast<std::int64_t>(r - start_row) * per_row;
  };

  // Descend the regular levels: advance while the next entry still fits, then
  // drop a level and repeat at a stride `fanout` times finer. At most `fanout`
  // comparisons per level, all contiguous within one array.
  std::size_t row = start_row;
  for (std::size_t lvl = level.size(); lvl-- > 0;) {
    auto const& lv    = level[lvl];
    auto const stride = stride_of(lvl);
    auto idx          = row / stride + 1;
    while (idx < lv.size()) {
      auto const candidate = std::min(idx * stride, num_rows);
      if (candidate <= row) { ++idx; continue; }
      if (cost(candidate) > budget) { break; }
      row = candidate;
      if (row >= num_rows) { return num_rows; }
      ++idx;
    }
  }
  return std::max(row, start_row + 1);
}

std::size_t variable_width_row_byte_skiplist::page_rows(std::size_t i) const noexcept
{
  if (i >= pages.size()) { return 0; }
  auto const end = (i + 1 < pages.size()) ? pages[i + 1].start_row : num_rows;
  return end - pages[i].start_row;
}

std::int64_t variable_width_row_byte_skiplist::page_bytes(std::size_t i) const noexcept
{
  if (i >= pages.size()) { return 0; }
  auto const end = (i + 1 < pages.size()) ? pages[i + 1].start_bytes : total_bytes();
  return end - pages[i].start_bytes;
}

std::size_t variable_width_row_byte_skiplist::find_page(std::size_t row) const noexcept
{
  if (pages.empty() || row >= num_rows) { return pages.size(); }
  // First page starting strictly after `row`; the covering page is the one before.
  auto it = std::upper_bound(pages.begin(), pages.end(), row,
                             [](std::size_t r, variable_width_page_boundary const& b) {
                               return r < b.start_row;
                             });
  if (it == pages.begin()) { return pages.size(); }
  return static_cast<std::size_t>(std::prev(it) - pages.begin());
}

variable_width_row_byte_skiplist build_row_byte_skiplist(
  std::vector<std::int32_t> const& offsets, std::size_t base_stride, std::size_t fanout,
  std::size_t page_size_bytes, std::size_t per_row_overhead_bytes)
{
  variable_width_row_byte_skiplist idx;
  if (offsets.size() < 2 || base_stride == 0 || fanout < 2) { return idx; }

  idx.base_stride = base_stride;
  idx.fanout      = fanout;
  idx.num_rows    = offsets.size() - 1;

  // Level 0, rebased on the column's first offset so values read as "bytes since
  // the start of this chunk" regardless of where the view sits in its parent.
  auto const first = static_cast<std::int64_t>(offsets.front());
  std::vector<std::int64_t> lv0;
  lv0.reserve(idx.num_rows / base_stride + 2);
  for (std::size_t row = 0; row < idx.num_rows; row += base_stride) {
    lv0.push_back(static_cast<std::int64_t>(offsets[row]) - first);
  }
  lv0.push_back(static_cast<std::int64_t>(offsets.back()) - first);  // pinned to num_rows
  if (!std::is_sorted(lv0.begin(), lv0.end())) { return variable_width_row_byte_skiplist{}; }
  idx.level.push_back(std::move(lv0));

  // Regular levels: each samples the one below by `fanout`.
  while (idx.level.back().size() > 2) {
    auto const& below = idx.level.back();
    std::vector<std::int64_t> up;
    up.reserve(below.size() / fanout + 2);
    for (std::size_t i = 0; i < below.size() - 1; i += fanout) { up.push_back(below[i]); }
    up.push_back(below.back());
    if (up.size() >= below.size()) { break; }  // no narrowing; stop rather than loop
    idx.level.push_back(std::move(up));
  }

  // Top level: walk once, recording every row where the running total reaches the
  // budget. This is the page directory -- no separate cutting pass exists.
  if (page_size_bytes > 0) {
    idx.page_size_bytes = page_size_bytes;
    auto const budget   = static_cast<std::int64_t>(page_size_bytes);
    std::size_t start   = 0;
    while (start < idx.num_rows) {
      idx.pages.push_back(
        variable_width_page_boundary{start, idx.bytes_before(start)});
      auto const end = idx.find_page_end(start, budget, per_row_overhead_bytes);
      start          = std::max(end, start + 1);
    }
  }
  return idx;
}

std::vector<std::size_t> cut_pages_with_skiplist(
  variable_width_row_byte_skiplist const& index, std::size_t page_size_bytes,
  std::size_t per_row_overhead_bytes)
{
  std::vector<std::size_t> starts;
  if (!index.valid() || page_size_bytes == 0) { return starts; }

  // Already cut for this budget -- just flatten the top level.
  if (!index.pages.empty() && index.page_size_bytes == page_size_bytes) {
    starts.reserve(index.pages.size());
    for (auto const& b : index.pages) { starts.push_back(b.start_row); }
    return starts;
  }
  // Different budget than the one baked in: cut on the fly.
  auto const budget = static_cast<std::int64_t>(page_size_bytes);
  std::size_t start = 0;
  while (start < index.num_rows) {
    starts.push_back(start);
    start = std::max(index.find_page_end(start, budget, per_row_overhead_bytes), start + 1);
  }
  return starts;
}

variable_width_row_byte_map build_row_byte_map(
  variable_width_offsets_extraction const& extraction)
{
  variable_width_row_byte_map map;
  if (!extraction.valid || extraction.sample_stride == 0 ||
      extraction.sampled_offsets.empty()) {
    return map;
  }
  auto const num_rows = static_cast<std::size_t>(extraction.col.size());
  if (num_rows == 0) { return map; }

  map.stride   = extraction.sample_stride;
  map.num_rows = num_rows;
  map.byte_prefix.reserve(extraction.sampled_offsets.size() + 1);
  // Rebase onto the column's first offset so the map reads as "bytes since the
  // start of this chunk", independent of where `col` sits in its parent buffer.
  for (auto const sampled : extraction.sampled_offsets) {
    map.byte_prefix.push_back(static_cast<std::int64_t>(sampled) -
                              static_cast<std::int64_t>(extraction.first_offset));
  }
  // Pin the final sample to num_rows exactly, so byte_prefix.back() is the
  // chunk's exact total chars bytes rather than an interpolated estimate.
  auto const total = static_cast<std::int64_t>(extraction.last_offset) -
                     static_cast<std::int64_t>(extraction.first_offset);
  if (map.byte_prefix.size() >= 2 &&
      (map.byte_prefix.size() - 1) * map.stride >= num_rows) {
    map.byte_prefix.back() = total;  // last sample already lands on num_rows
  } else {
    map.byte_prefix.push_back(total);
  }
  // A non-monotone prefix would break both the binary search and the
  // interpolation; a truncated/failed copy is the only way to get one, and
  // falling back to average-based cutting is strictly safer than trusting it.
  if (!std::is_sorted(map.byte_prefix.begin(), map.byte_prefix.end())) {
    return variable_width_row_byte_map{};
  }
  return map;
}

variable_width_offsets_extraction queue_variable_width_offsets_extraction(
  cudf::column_view const& col, rmm::cuda_stream_view stream, bool full_offsets)
{
  if (col.type().id() != cudf::type_id::STRING || col.size() <= 0) {
    return variable_width_offsets_extraction{};
  }

  auto const num_rows = col.size();
  cudf::strings_column_view scv(col);
  auto const* offsets_data = scv.offsets().data<int32_t>() + col.offset();

  variable_width_offsets_extraction extraction;
  extraction.valid = true;
  extraction.col   = col;

  // Only the first and last offset -- their difference is the column's exact
  // total chars byte size (see the header comment for why this suffices).
  // Same offsets-child layout debug_utils.cpp's STRING cell-extraction path
  // already relies on: dtype fixed at int32_t, col.offset() accounts for
  // `col` itself being a view into a larger buffer. Async and NOT
  // synchronized here on purpose -- callers batch several columns' copies
  // behind one synchronize().
  cudaMemcpyAsync(&extraction.first_offset,
                  offsets_data,
                  sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  cudaMemcpyAsync(&extraction.last_offset,
                  offsets_data + num_rows,
                  sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());

  // Plus a strided sample of the offsets in between -- the map's raw material.
  // One cudaMemcpy2DAsync with a 4-byte width and a stride-sized source pitch
  // picks up every sample_stride'th offset in a single API call, no kernel and
  // no gather map. Measured on this box against a 10M-row chunk: 7.5us at 512
  // samples, 17.7us at 4096, versus ~6ms to copy every offset -- which is the
  // cost that made exact per-row byte budgeting untenable in the first place.
  // Still async and unsynchronized, so it batches behind the same single
  // synchronize() the caller already pays for first/last.
  auto const rows = static_cast<std::size_t>(num_rows);
  if (full_offsets) {
    // Exact path: one contiguous copy of the whole offsets child (rows + 1
    // entries). Still stream-ordered behind the caller's single synchronize().
    extraction.full_offsets.resize(rows + 1);
    if (cudaMemcpyAsync(extraction.full_offsets.data(),
                        offsets_data,
                        (rows + 1) * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost,
                        stream.value()) == cudaSuccess) {
      return extraction;
    }
    extraction.full_offsets.clear();  // fall through to the strided sample
  }
  auto const stride =
    std::max<std::size_t>(1, (rows + kOffsetSampleBudget - 1) / kOffsetSampleBudget);
  // (samples - 1) * stride <= rows keeps the last sampled element inside the
  // offsets child, which has rows + 1 entries. Overrunning it makes the whole
  // 2D copy fail and silently leave the destination untouched.
  auto const samples = rows / stride + 1;
  if (samples >= 2) {
    extraction.sampled_offsets.resize(samples);
    auto const status = cudaMemcpy2DAsync(extraction.sampled_offsets.data(),
                                          sizeof(std::int32_t),
                                          offsets_data,
                                          stride * sizeof(std::int32_t),
                                          sizeof(std::int32_t),
                                          samples,
                                          cudaMemcpyDeviceToHost,
                                          stream.value());
    if (status == cudaSuccess) {
      extraction.sample_stride = stride;
    } else {
      // Leave sample_stride at 0; build_row_byte_map then returns an invalid
      // map and page cutting falls back to the average-based path.
      extraction.sampled_offsets.clear();
    }
  }
  return extraction;
}

variable_width_chunk_page_index build_variable_width_column_pages(
  variable_width_offsets_extraction const& extraction, std::size_t page_size_bytes,
  std::size_t chunk_index, std::string const& table_name, std::string const& column_name,
  cucascade::memory::memory_space& memory_space, rmm::cuda_stream_view stream,
  std::size_t alignment_rows, std::size_t slab_max_rows)
{
  if (!extraction.valid) { return variable_width_chunk_page_index{}; }
  auto const& col      = extraction.col;
  auto const num_rows  = static_cast<std::size_t>(col.size());
  if (num_rows == 0 || page_size_bytes == 0) { return variable_width_chunk_page_index{}; }

  // Preferred path: cut against the sparse row->byte map, so every page's real
  // footprint (chars + offsets child) lands within page_size_bytes regardless of
  // how row lengths are distributed inside the chunk.
  // Fixed-slab path: every page gets identically-sized chars and offsets
  // allocations, which is what takes external fragmentation to zero (see
  // materialize_page_into_slab). Requires exact per-row offsets, since the cut
  // must guarantee the chars run fits the slab rather than merely approximate it.
  if (slab_max_rows > 0) {
    // Page boundaries come from one batched binary search on the offsets already
    // resident on the device -- the offsets child IS the prefix sum, so nothing is
    // summed, scanned, or copied to the host. Bytes-per-row for the size class is
    // read from the two ends of the run, also O(1).
    auto const total_bytes =
      static_cast<std::int64_t>(extraction.last_offset - extraction.first_offset);
    auto const bytes_per_row =
      num_rows > 0 ? static_cast<double>(total_bytes) / static_cast<double>(num_rows) : 1.0;
    auto const chars_slab_bytes = choose_chars_slab_bytes(
      bytes_per_row, slab_max_rows, /*min_bytes=*/std::size_t{1} << 20, page_size_bytes);
    auto const starts =
      find_page_starts_on_device(col, chars_slab_bytes, slab_max_rows, stream);
    if (!starts.empty()) {
      auto const offsets_slab_bytes = (slab_max_rows + 1) * sizeof(std::int32_t);
      variable_width_chunk_page_index slab_idx;
      bool ok = true;
      for (std::size_t p = 0; p < starts.size() && ok; ++p) {
        auto const start_row  = starts[p];
        auto const next_start = (p + 1 < starts.size()) ? starts[p + 1] : num_rows;
        auto const page_rows  = next_start - start_row;
        if (page_rows == 0) { continue; }
        auto owned = materialize_page_into_slab(
          col, start_row, page_rows, chars_slab_bytes, offsets_slab_bytes, memory_space, stream);
        if (!owned) { ok = false; break; }  // nullable / oversized: fall back below
        variable_width_column_page page;
        page.page_index   = slab_idx.pages.size();
        page.start_row    = start_row;
        page.num_rows     = page_rows;
        page.table_name   = table_name;
        page.column_name  = column_name;
        page.chunk_index  = chunk_index;
        page.memory_space = &memory_space;
        page.owned_column = std::move(owned);
        // Uniform by construction, so this is the same number for every page --
        // that identity is the property the slab layout exists to provide.
        page.num_bytes = static_cast<std::size_t>(page.owned_column->alloc_size());
        slab_idx.page_start_rows.push_back(start_row);
        slab_idx.pages.push_back(std::move(page));
      }
      if (ok && !slab_idx.pages.empty()) { return slab_idx; }
    }
  }

  // Exact path: when the caller asked for full offsets, cut from the
  // skiplist so page boundaries land on the exact budget rather than an
  // interpolated estimate.
  auto row_byte_map = build_row_byte_map(extraction);
  std::vector<std::size_t> page_starts;
  if (alignment_rows == 0 && !extraction.full_offsets.empty()) {
    auto const skip = build_row_byte_skiplist(extraction.full_offsets, 1, 32, page_size_bytes,
                                             kVariableWidthOffsetsOverheadPerRow);
    page_starts =
      cut_pages_with_skiplist(skip, page_size_bytes, kVariableWidthOffsetsOverheadPerRow);
  }
  if (!page_starts.empty()) {
    // cut from the skiplist above
  } else if (alignment_rows > 0) {
    // Cut on the read grid so a range maps to exactly one page -- see the header.
    for (std::size_t start_row = 0; start_row < num_rows; start_row += alignment_rows) {
      page_starts.push_back(start_row);
    }
  } else {
    page_starts =
      cut_pages_by_byte_budget(row_byte_map, page_size_bytes, kVariableWidthOffsetsOverheadPerRow);
  }

  if (page_starts.empty()) {
    // Fallback for a chunk whose strided sample never made it back: a flat
    // rows-per-page cut from the column average. Still charges the offsets
    // child's per-row cost, which the average alone used to ignore.
    auto const total_bytes =
      static_cast<std::size_t>(extraction.last_offset - extraction.first_offset);
    auto const avg_bytes_per_row =
      std::max<std::size_t>(1, total_bytes / num_rows + kVariableWidthOffsetsOverheadPerRow);
    auto const rows_per_page = std::max<std::size_t>(1, page_size_bytes / avg_bytes_per_row);
    for (std::size_t start_row = 0; start_row < num_rows; start_row += rows_per_page) {
      page_starts.push_back(start_row);
    }
  }

  variable_width_chunk_page_index idx;
  idx.row_byte_map = std::move(row_byte_map);
  for (std::size_t p = 0; p < page_starts.size(); ++p) {
    auto const start_row = page_starts[p];
    auto const next_start = (p + 1 < page_starts.size()) ? page_starts[p + 1] : num_rows;
    auto const page_rows  = next_start - start_row;
    if (page_rows == 0) { continue; }

    auto const begin = static_cast<cudf::size_type>(start_row);
    auto const end   = static_cast<cudf::size_type>(start_row + page_rows);
    auto views        = cudf::slice(col, {begin, end});
    if (views.empty()) { continue; }

    variable_width_column_page page;
    page.page_index   = idx.pages.size();
    page.start_row    = start_row;
    page.num_rows     = page_rows;
    page.table_name   = table_name;
    page.column_name  = column_name;
    page.chunk_index  = chunk_index;
    page.memory_space = &memory_space;
    // Copy-construct (not just view) so the page owns independent device
    // storage -- the same slice-then-copy pattern
    // fixed_page_databatch_provider::materialize_owned_subrange uses, which
    // is generic cudf::column machinery and needs no string-specific code:
    // the copy constructor deep-copies the offsets and chars children too.
    page.owned_column = std::make_shared<cudf::column>(
      views.front(), stream, memory_space.get_default_allocator());
    // Real allocated bytes (offsets child included), a host-tracked property
    // of the now-constructed column -- no extra device sync needed. Matches
    // fixed_width_page_resident_alloc_bytes's precedent for fixed-width pages.
    page.num_bytes = static_cast<std::size_t>(page.owned_column->alloc_size());

    idx.page_start_rows.push_back(start_row);
    idx.pages.push_back(std::move(page));
  }
  return idx;
}

variable_width_chunk_page_index index_variable_width_column_pages_from_view(
  cudf::column_view const& col, std::size_t page_size_bytes, std::size_t chunk_index,
  std::string const& table_name, std::string const& column_name,
  cucascade::memory::memory_space& memory_space, rmm::cuda_stream_view stream)
{
  auto extraction = queue_variable_width_offsets_extraction(col, stream);
  if (!extraction.valid) { return variable_width_chunk_page_index{}; }
  stream.synchronize();
  return build_variable_width_column_pages(
    extraction, page_size_bytes, chunk_index, table_name, column_name, memory_space, stream);
}

void finalize_page_directory(variable_width_row_byte_skiplist& index,
                             std::size_t chars_budget_bytes,
                             std::size_t max_rows_per_page)
{
  index.pages.clear();
  index.page_size_bytes = 0;
  if (!index.valid() || chars_budget_bytes == 0 || max_rows_per_page == 0) { return; }

  index.page_size_bytes = chars_budget_bytes;
  auto const budget     = static_cast<std::int64_t>(chars_budget_bytes);
  std::size_t start     = 0;
  while (start < index.num_rows) {
    index.pages.push_back(
      variable_width_page_boundary{start, index.bytes_before(start)});
    // Chars-only budget: the offsets child lives in its own slab, bounded by the
    // row cap rather than by these bytes.
    auto end = index.find_page_end(start, budget, /*per_row_overhead=*/0);
    end      = std::min(end, start + max_rows_per_page);
    start    = std::max(end, start + 1);
  }
}

std::vector<std::size_t> find_page_starts_on_device(cudf::column_view const& col,
                                                    std::size_t chars_budget_bytes,
                                                    std::size_t max_rows_per_page,
                                                    rmm::cuda_stream_view stream)
{
  std::vector<std::size_t> starts;
  if (col.type().id() != cudf::type_id::STRING || col.size() <= 0 ||
      chars_budget_bytes == 0 || max_rows_per_page == 0) {
    return starts;
  }
  auto const num_rows = static_cast<std::size_t>(col.size());

  cudf::strings_column_view scv(col);
  auto const offsets_view = scv.offsets();
  if (offsets_view.type().id() != cudf::type_id::INT32) { return starts; }
  auto const* offsets_base = offsets_view.data<std::int32_t>() + col.offset();

  // The chunk's byte span, from the two ends of its offsets run. O(1).
  std::int32_t first_offset = 0;
  std::int32_t last_offset  = 0;
  cudaMemcpyAsync(&first_offset, offsets_base, sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost, stream.value());
  cudaMemcpyAsync(&last_offset, offsets_base + num_rows, sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();

  auto const total_bytes = static_cast<std::int64_t>(last_offset) -
                           static_cast<std::int64_t>(first_offset);
  if (total_bytes <= 0) { return {0}; }

  // Search targets: one per page boundary, on an even byte grid.
  auto const page_count = static_cast<std::size_t>(
    (total_bytes + static_cast<std::int64_t>(chars_budget_bytes) - 1) /
    static_cast<std::int64_t>(chars_budget_bytes));
  if (page_count <= 1) { return {0}; }

  std::vector<std::int32_t> targets;
  targets.reserve(page_count - 1);
  for (std::size_t k = 1; k < page_count; ++k) {
    targets.push_back(static_cast<std::int32_t>(
      static_cast<std::int64_t>(first_offset) +
      static_cast<std::int64_t>(k) * static_cast<std::int64_t>(chars_budget_bytes)));
  }

  // One batched lower_bound over the offsets run already resident on the device.
  auto const haystack = cudf::column_view{offsets_view.type(),
                                          static_cast<cudf::size_type>(num_rows + 1),
                                          offsets_base,
                                          nullptr,
                                          0};
  auto needle_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, static_cast<cudf::size_type>(targets.size()),
    cudf::mask_state::UNALLOCATED, stream);
  cudaMemcpyAsync(needle_col->mutable_view().data<std::int32_t>(), targets.data(),
                  targets.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                  stream.value());

  auto positions = cudf::lower_bound(cudf::table_view{{haystack}},
                                     cudf::table_view{{needle_col->view()}},
                                     {cudf::order::ASCENDING},
                                     {cudf::null_order::BEFORE},
                                     stream);
  std::vector<std::int32_t> host_pos(targets.size());
  cudaMemcpyAsync(host_pos.data(), positions->view().data<std::int32_t>(),
                  host_pos.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();

  // Assemble, enforcing strict growth and the row cap (which bounds the page's
  // offsets buffer independently of its chars bytes).
  starts.push_back(0);
  for (auto pos : host_pos) {
    auto next = static_cast<std::size_t>(std::max(pos, 0));
    next      = std::min(next, num_rows);
    auto const capped = starts.back() + max_rows_per_page;
    if (next > capped) { next = capped; }
    if (next <= starts.back()) { continue; }
    if (next >= num_rows) { break; }
    starts.push_back(next);
  }
  // The row cap can bind before the byte grid does; keep cutting if rows remain.
  while (starts.back() + max_rows_per_page < num_rows) {
    starts.push_back(starts.back() + max_rows_per_page);
  }
  return starts;
}

std::vector<std::size_t> cut_pages_for_slabs(variable_width_row_byte_skiplist const& index,
                                             std::size_t chars_budget_bytes,
                                             std::size_t max_rows_per_page)
{
  std::vector<std::size_t> starts;
  if (!index.valid() || chars_budget_bytes == 0 || max_rows_per_page == 0) { return starts; }

  // Already finalized for this budget: the top level IS the answer, just flatten it.
  if (!index.pages.empty() && index.page_size_bytes == chars_budget_bytes) {
    starts.reserve(index.pages.size());
    for (auto const& b : index.pages) { starts.push_back(b.start_row); }
    return starts;
  }
  auto const budget = static_cast<std::int64_t>(chars_budget_bytes);
  std::size_t start = 0;
  while (start < index.num_rows) {
    starts.push_back(start);
    auto end = index.find_page_end(start, budget, /*per_row_overhead=*/0);
    end      = std::min(end, start + max_rows_per_page);
    start    = std::max(end, start + 1);
  }
  return starts;
}

std::size_t choose_chars_slab_bytes(double bytes_per_row, std::size_t max_rows_per_page,
                                    std::size_t min_bytes, std::size_t max_bytes)
{
  if (max_rows_per_page == 0 || min_bytes == 0) { return max_bytes; }
  auto const wanted =
    static_cast<double>(max_rows_per_page) * std::max(bytes_per_row, 1.0);
  // Smallest power-of-two class that holds a full page of this column's rows.
  // Rows that overrun it simply end the page early (cut_pages_for_slabs enforces
  // the same budget), so an under-estimate costs page count, never correctness.
  std::size_t slab = min_bytes;
  while (slab < max_bytes && static_cast<double>(slab) < wanted) { slab *= 2; }
  return std::min(slab, max_bytes);
}

std::shared_ptr<cudf::column> materialize_page_into_slab(
  cudf::column_view const& col, std::size_t start_row, std::size_t num_rows,
  std::size_t chars_slab_bytes, std::size_t offsets_slab_bytes,
  cucascade::memory::memory_space& memory_space, rmm::cuda_stream_view stream)
{
  if (col.type().id() != cudf::type_id::STRING || col.size() <= 0 || num_rows == 0) {
    return nullptr;
  }
  if (start_row + num_rows > static_cast<std::size_t>(col.size())) { return nullptr; }
  // Nullable columns would need a sliced null mask copied into a third fixed
  // allocation; out of scope for the prototype, so they stay on the exact path.
  if (col.nullable() && col.null_count() > 0) { return nullptr; }

  auto const offsets_needed = (num_rows + 1) * sizeof(std::int32_t);
  if (offsets_slab_bytes < offsets_needed) { return nullptr; }

  cudf::strings_column_view scv(col);
  auto const offsets_view = scv.offsets();
  if (offsets_view.type().id() != cudf::type_id::INT32) { return nullptr; }
  auto const* offsets_base = offsets_view.data<std::int32_t>() + col.offset();

  // The page's own offsets run, still absolute. cudf::slice is a view, no copy.
  auto const begin = static_cast<cudf::size_type>(start_row);
  auto const end   = static_cast<cudf::size_type>(start_row + num_rows + 1);
  auto const slices =
    cudf::slice(cudf::column_view{offsets_view.type(),
                                  offsets_view.size() - col.offset(),
                                  offsets_base,
                                  nullptr,
                                  0},
                {begin, end});
  if (slices.empty()) { return nullptr; }

  // First/last offsets bound the chars run. Read them back so the copy sizes and
  // the rebase constant are known on the host.
  std::int32_t first_offset = 0;
  std::int32_t last_offset  = 0;
  cudaMemcpyAsync(&first_offset, offsets_base + start_row, sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost, stream.value());
  cudaMemcpyAsync(&last_offset, offsets_base + start_row + num_rows, sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();

  auto const chars_bytes = static_cast<std::size_t>(last_offset - first_offset);
  if (chars_bytes > chars_slab_bytes) { return nullptr; }  // caller mis-sized the page

  auto mr = memory_space.get_default_allocator();

  // Rebase the offsets so the page's first row starts at 0. binary_operation
  // allocates its own (small, transient) result; it is copied into the fixed
  // slab and released immediately, so it never joins the cache's live set.
  cudf::numeric_scalar<std::int32_t> base_scalar(first_offset, true, stream);
  auto rebased = cudf::binary_operation(slices.front(),
                                        base_scalar,
                                        cudf::binary_operator::SUB,
                                        cudf::data_type{cudf::type_id::INT32},
                                        stream,
                                        mr);
  if (!rebased || rebased->size() != static_cast<cudf::size_type>(num_rows + 1)) {
    return nullptr;
  }

  // The two fixed-size allocations. Every page in the cache asks for exactly
  // these byte counts, which is what drives external fragmentation to zero.
  rmm::device_buffer offsets_slab{offsets_slab_bytes, stream, mr};
  rmm::device_buffer chars_slab{chars_slab_bytes, stream, mr};
  cudaMemcpyAsync(offsets_slab.data(),
                  rebased->view().data<std::int32_t>(),
                  offsets_needed,
                  cudaMemcpyDeviceToDevice,
                  stream.value());
  if (chars_bytes > 0) {
    cudaMemcpyAsync(chars_slab.data(),
                    scv.chars_begin(stream) + first_offset,
                    chars_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream.value());
  }

  // The offsets column reports num_rows + 1 elements while owning a larger
  // buffer; cudf only ever reads the first size() entries, so the unused tail is
  // simply internal fragmentation.
  auto offsets_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                       static_cast<cudf::size_type>(num_rows + 1),
                                                       std::move(offsets_slab),
                                                       rmm::device_buffer{},
                                                       0);
  auto page_column = cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                               std::move(offsets_column),
                                               std::move(chars_slab),
                                               0,
                                               rmm::device_buffer{});
  return std::shared_ptr<cudf::column>{std::move(page_column)};
}

variable_width_column_page const* find_covering_variable_page(
  variable_width_chunk_page_index const& idx, std::size_t row)
{
  if (idx.pages.empty()) { return nullptr; }

  // upper_bound finds the first start_row strictly greater than `row`; the
  // covering page is the one just before that. page_start_rows[0] == 0 by
  // construction, so this is only ever begin() (no covering page) if the
  // index itself is malformed -- kept as a defensive check, not a case that
  // occurs for any row from index_variable_width_column_pages's own output.
  auto it = std::upper_bound(idx.page_start_rows.begin(), idx.page_start_rows.end(), row);
  if (it == idx.page_start_rows.begin()) { return nullptr; }

  auto const page_pos = static_cast<std::size_t>(std::prev(it) - idx.page_start_rows.begin());
  auto const& page     = idx.pages[page_pos];
  if (page.state != variable_width_page_state::resident) { return nullptr; }
  if (row >= page.start_row + page.num_rows) { return nullptr; }  // past this chunk's last row
  return &page;
}

bool variable_width_page_is_active(variable_width_column_page const& page)
{
  return page.active_reader_count.load(std::memory_order_relaxed) != 0;
}

void adjust_variable_width_page_active_reader(variable_width_column_page const& page, int delta)
{
  if (delta > 0) {
    page.active_reader_count.fetch_add(static_cast<std::uint32_t>(delta), std::memory_order_relaxed);
  } else if (delta < 0) {
    page.active_reader_count.fetch_sub(static_cast<std::uint32_t>(-delta), std::memory_order_relaxed);
  }
}

namespace {
std::atomic<std::uint64_t> g_shared_page_lru_tick{1};
}  // namespace

std::uint64_t next_shared_page_lru_tick()
{
  return g_shared_page_lru_tick.fetch_add(1, std::memory_order_relaxed) + 1;
}

void touch_variable_width_page(variable_width_column_page const& page)
{
  page.last_access_tick.store(next_shared_page_lru_tick(), std::memory_order_relaxed);
}

}  // namespace sirius::scan_manager
