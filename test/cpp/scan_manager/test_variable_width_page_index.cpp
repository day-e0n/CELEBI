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

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "scan_manager/variable_width_page_index.hpp"
#include "utils/data_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace sirius::scan_manager;
namespace test_utils = sirius::test::operator_utils;

namespace {

/// Ground truth for find_covering_variable_page: a plain linear scan, kept
/// deliberately dumb so it can't share a bug with the binary-search version
/// it's cross-checked against.
std::size_t linear_scan_covering_page(std::vector<std::size_t> const& row_byte_lengths,
                                       std::size_t page_size_bytes, std::size_t target_row)
{
  std::size_t row = 0, page = 0;
  while (row < row_byte_lengths.size()) {
    std::size_t bytes = 0, num_rows = 0;
    std::size_t const start_row = row;
    while (row < row_byte_lengths.size()) {
      std::size_t const next_len = row_byte_lengths[row];
      if (num_rows > 0 && bytes + next_len > page_size_bytes) { break; }
      bytes += next_len;
      ++num_rows;
      ++row;
      if (num_rows == 1 && bytes > page_size_bytes) { break; }
    }
    if (target_row >= start_row && target_row < start_row + num_rows) { return page; }
    ++page;
  }
  return static_cast<std::size_t>(-1);  // not found
}

/// Reads a STRING column's values back to host, the same offsets+chars
/// device->host copy pattern debug_utils.cpp's STRING cell extraction (and
/// index_variable_width_column_pages_from_view itself) uses -- an
/// independent readback path so this test can't share a bug with the
/// implementation it's checking.
std::vector<std::string> host_strings_from_column(cudf::column_view const& col,
                                                   rmm::cuda_stream_view stream)
{
  cudf::strings_column_view scv(col);
  auto const num_rows    = col.size();
  auto const num_offsets = static_cast<std::size_t>(num_rows) + 1;

  std::vector<int32_t> host_offsets(num_offsets);
  cudaMemcpyAsync(host_offsets.data(),
                  scv.offsets().data<int32_t>() + col.offset(),
                  num_offsets * sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();

  auto const chars_start = host_offsets[0];
  auto const chars_bytes = host_offsets[num_rows] - chars_start;
  std::vector<char> host_chars(static_cast<std::size_t>(chars_bytes));
  if (chars_bytes > 0) {
    cudaMemcpyAsync(host_chars.data(),
                    scv.chars_begin(stream) + chars_start,
                    static_cast<std::size_t>(chars_bytes),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();
  }

  std::vector<std::string> out(static_cast<std::size_t>(num_rows));
  for (cudf::size_type r = 0; r < num_rows; ++r) {
    auto const start = host_offsets[r] - chars_start;
    auto const end    = host_offsets[r + 1] - chars_start;
    out[static_cast<std::size_t>(r)] = std::string(host_chars.data() + start, end - start);
  }
  return out;
}

}  // namespace

TEST_CASE("index_variable_width_column_pages: hand-traced boundaries", "[variable_width_page_index]")
{
  // 100-byte budget; row 4 (200 bytes) is deliberately oversized.
  std::vector<std::size_t> lens = {30, 30, 50, 10, 200, 5, 5};
  auto idx                      = index_variable_width_column_pages(lens, 100);

  REQUIRE(idx.pages.size() == 4);
  CHECK(idx.pages[0].start_row == 0);
  CHECK(idx.pages[0].num_rows == 2);
  CHECK(idx.pages[0].num_bytes == 60);
  CHECK(idx.pages[1].start_row == 2);
  CHECK(idx.pages[1].num_rows == 2);
  CHECK(idx.pages[1].num_bytes == 60);
  CHECK(idx.pages[2].start_row == 4);
  CHECK(idx.pages[2].num_rows == 1);
  CHECK(idx.pages[2].num_bytes == 200);  // oversized row, exceeds budget on its own
  CHECK(idx.pages[3].start_row == 5);
  CHECK(idx.pages[3].num_rows == 2);
  CHECK(idx.pages[3].num_bytes == 10);

  CHECK(idx.page_start_rows == std::vector<std::size_t>{0, 2, 4, 5});
}

TEST_CASE("index_variable_width_column_pages: every row is covered exactly once", "[variable_width_page_index]")
{
  std::vector<std::size_t> lens = {30, 30, 50, 10, 200, 5, 5, 1, 99, 100, 101, 40};
  auto idx                      = index_variable_width_column_pages(lens, 100);

  std::size_t total_rows = 0;
  for (std::size_t p = 0; p < idx.pages.size(); ++p) {
    auto const& page = idx.pages[p];
    CHECK(page.page_index == p);
    CHECK(page.start_row == total_rows);  // pages are contiguous, no gaps or overlaps
    total_rows += page.num_rows;
    if (page.num_rows > 1) { CHECK(page.num_bytes <= 100); }  // budget honored unless lone oversized row
  }
  CHECK(total_rows == lens.size());
}

namespace {

/// Builds a map the way build_row_byte_map would, but straight from known
/// per-row lengths -- so a test can state the ground truth in row lengths and
/// still exercise the real sampling/interpolation arithmetic.
variable_width_row_byte_map make_map(std::vector<std::size_t> const& row_lengths,
                                     std::size_t stride)
{
  variable_width_row_byte_map map;
  map.stride   = stride;
  map.num_rows = row_lengths.size();
  std::vector<std::int64_t> exact(row_lengths.size() + 1, 0);
  for (std::size_t i = 0; i < row_lengths.size(); ++i) {
    exact[i + 1] = exact[i] + static_cast<std::int64_t>(row_lengths[i]);
  }
  for (std::size_t row = 0; row < row_lengths.size(); row += stride) {
    map.byte_prefix.push_back(exact[row]);
  }
  if (map.byte_prefix.empty() || (map.byte_prefix.size() - 1) * stride < row_lengths.size()) {
    map.byte_prefix.push_back(exact.back());
  } else {
    map.byte_prefix.back() = exact.back();
  }
  return map;
}

std::int64_t exact_bytes_in_range(std::vector<std::size_t> const& row_lengths,
                                  std::size_t begin, std::size_t end)
{
  std::int64_t total = 0;
  for (std::size_t i = begin; i < end && i < row_lengths.size(); ++i) {
    total += static_cast<std::int64_t>(row_lengths[i]);
  }
  return total;
}

}  // namespace

TEST_CASE("variable_width_row_byte_map: exact at sample rows, bounded error between",
          "[variable_width_page_index]")
{
  // Deliberately skewed: long rows bunched at the front of each bucket, so
  // interpolation inside a bucket is as wrong as it can be for this stride.
  std::vector<std::size_t> lengths;
  for (std::size_t i = 0; i < 1000; ++i) { lengths.push_back((i % 100) < 10 ? 200 : 5); }
  constexpr std::size_t stride = 64;
  auto const map               = make_map(lengths, stride);
  REQUIRE(map.valid());

  // Exact total, and exact at every sample boundary.
  REQUIRE(map.total_bytes() == exact_bytes_in_range(lengths, 0, lengths.size()));
  for (std::size_t row = 0; row <= lengths.size(); row += stride) {
    REQUIRE(map.bytes_before(row) == exact_bytes_in_range(lengths, 0, row));
  }

  // In between, the error can never exceed one bucket's byte span.
  for (std::size_t row = 0; row < lengths.size(); ++row) {
    auto const bucket_lo    = (row / stride) * stride;
    auto const bucket_hi    = std::min(bucket_lo + stride, lengths.size());
    auto const bucket_bytes = exact_bytes_in_range(lengths, bucket_lo, bucket_hi);
    auto const error = std::abs(map.bytes_before(row) - exact_bytes_in_range(lengths, 0, row));
    REQUIRE(error <= bucket_bytes);
  }
}

TEST_CASE("variable_width_row_byte_map: degenerate cases", "[variable_width_page_index]")
{
  variable_width_row_byte_map empty;
  REQUIRE_FALSE(empty.valid());
  REQUIRE(empty.total_bytes() == 0);
  REQUIRE(empty.bytes_before(42) == 0);
  REQUIRE(empty.bytes_in_range(0, 42) == 0);

  auto const map = make_map(std::vector<std::size_t>(10, 4), 4);
  REQUIRE(map.bytes_in_range(5, 5) == 0);
  REQUIRE(map.bytes_in_range(7, 3) == 0);           // reversed range
  REQUIRE(map.bytes_before(1000) == map.total_bytes());  // clamped past the end
}

TEST_CASE("cut_pages_by_byte_budget: every page fits the budget and rows are covered once",
          "[variable_width_page_index]")
{
  std::mt19937 rng{20260818};
  std::uniform_int_distribution<std::size_t> len_dist{1, 120};
  std::vector<std::size_t> lengths(5000);
  for (auto& l : lengths) { l = len_dist(rng); }

  constexpr std::size_t stride       = 32;
  constexpr std::size_t budget       = 4096;
  constexpr std::size_t per_row_cost = 4;
  auto const map    = make_map(lengths, stride);
  auto const starts = cut_pages_by_byte_budget(map, budget, per_row_cost);

  REQUIRE_FALSE(starts.empty());
  REQUIRE(starts.front() == 0);
  for (std::size_t i = 1; i < starts.size(); ++i) { REQUIRE(starts[i] > starts[i - 1]); }

  // Every page's real cost stays within budget -- allowing one bucket of
  // interpolation slack, plus the single-oversized-row escape hatch.
  for (std::size_t i = 0; i < starts.size(); ++i) {
    auto const begin = starts[i];
    auto const end   = (i + 1 < starts.size()) ? starts[i + 1] : lengths.size();
    REQUIRE(end > begin);
    auto const cost = exact_bytes_in_range(lengths, begin, end) +
                      static_cast<std::int64_t>(end - begin) * per_row_cost;
    if (end - begin == 1) { continue; }  // a lone row may legitimately exceed budget
    auto const bucket_lo    = (begin / stride) * stride;
    auto const bucket_hi    = std::min(bucket_lo + stride, lengths.size());
    auto const slack        = exact_bytes_in_range(lengths, bucket_lo, bucket_hi) +
                       static_cast<std::int64_t>(stride) * per_row_cost;
    REQUIRE(cost <= static_cast<std::int64_t>(budget) + slack);
  }
  // ...and the pages tile the chunk exactly once.
  REQUIRE((starts.size() == 1 ? lengths.size() : starts[1]) > 0);
}

TEST_CASE("cut_pages_by_byte_budget: narrow columns still split, offsets overhead included",
          "[variable_width_page_index]")
{
  // A 1-char column is the case that used to degenerate to a single page: the
  // chars average is 1 byte/row, so a budget divided by that asks for far more
  // rows than exist. The real per-row cost is 1 + 4 offset bytes.
  std::vector<std::size_t> lengths(100'000, 1);
  auto const map    = make_map(lengths, 512);
  auto const starts = cut_pages_by_byte_budget(map, 64 * 1024, kVariableWidthOffsetsOverheadPerRow);

  // 64KiB / 5 bytes per row ~= 13k rows per page, so ~8 pages -- not 1.
  REQUIRE(starts.size() > 5);
  REQUIRE(starts.size() < 12);
  // Ignoring the offsets child undercounts the real footprint 5x, so it cuts
  // ~5x too few pages -- each of which then blows through the budget on the GPU.
  auto const chars_only = cut_pages_by_byte_budget(map, 64 * 1024, 0);
  REQUIRE(chars_only.size() == 2);
  REQUIRE(starts.size() >= 4 * chars_only.size());
}

TEST_CASE("cut_pages_by_byte_budget: a single oversized row cannot stall the cursor",
          "[variable_width_page_index]")
{
  std::vector<std::size_t> lengths{10, 10, 5000, 10, 10};
  auto const map    = make_map(lengths, 1);
  auto const starts = cut_pages_by_byte_budget(map, 100, 0);
  REQUIRE(starts.size() >= 3);
  REQUIRE(starts.back() < lengths.size());
  for (std::size_t i = 1; i < starts.size(); ++i) { REQUIRE(starts[i] > starts[i - 1]); }
}

TEST_CASE("cut_pages_by_byte_budget: invalid map and zero budget yield no pages",
          "[variable_width_page_index]")
{
  auto const map = make_map(std::vector<std::size_t>(100, 8), 16);
  REQUIRE(cut_pages_by_byte_budget(map, 0, 4).empty());
  REQUIRE(cut_pages_by_byte_budget(variable_width_row_byte_map{}, 1024, 4).empty());
}

namespace {

/// Raw offsets child for a column with the given per-row byte lengths.
std::vector<std::int32_t> offsets_from_lengths(std::vector<std::size_t> const& lengths,
                                               std::int32_t base = 0)
{
  std::vector<std::int32_t> off;
  off.reserve(lengths.size() + 1);
  std::int32_t acc = base;
  off.push_back(acc);
  for (auto l : lengths) { acc += static_cast<std::int32_t>(l); off.push_back(acc); }
  return off;
}

}  // namespace

TEST_CASE("row_byte_skiplist: top level IS the page directory", "[variable_width_page_index]")
{
  std::mt19937 rng{4242};
  std::uniform_int_distribution<std::size_t> len{5, 120};
  std::vector<std::size_t> lengths(50000);
  for (auto& l : lengths) { l = len(rng); }

  constexpr std::size_t budget       = 64 * 1024;
  constexpr std::size_t per_row_cost = kVariableWidthOffsetsOverheadPerRow;
  auto const idx =
    build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 16, budget, per_row_cost);
  REQUIRE(idx.valid());
  REQUIRE(idx.page_count() > 1);
  REQUIRE(idx.page_size_bytes == budget);

  // The directory tiles the chunk exactly once, in order.
  REQUIRE(idx.pages.front().start_row == 0);
  REQUIRE(idx.pages.front().start_bytes == 0);
  std::size_t covered = 0;
  for (std::size_t i = 0; i < idx.page_count(); ++i) {
    REQUIRE(idx.page_rows(i) > 0);
    covered += idx.page_rows(i);
    if (i + 1 < idx.page_count()) {
      REQUIRE(idx.pages[i + 1].start_row > idx.pages[i].start_row);
    }
  }
  REQUIRE(covered == lengths.size());

  // Every page's own byte cost is a difference of two stored values and, except
  // for a lone oversized row, fits the budget it was cut for.
  for (std::size_t i = 0; i < idx.page_count(); ++i) {
    auto const b = idx.pages[i].start_row;
    auto const e = b + idx.page_rows(i);
    std::int64_t truth = 0;
    for (std::size_t r = b; r < e; ++r) { truth += static_cast<std::int64_t>(lengths[r]); }
    REQUIRE(idx.page_bytes(i) == truth);
    if (idx.page_rows(i) > 1) {
      REQUIRE(truth + static_cast<std::int64_t>(idx.page_rows(i) * per_row_cost) <=
              static_cast<std::int64_t>(budget));
    }
  }

  // find_page agrees with a linear scan over the directory, for every row.
  for (std::size_t row = 0; row < lengths.size(); row += 7) {
    std::size_t expect = 0;
    while (expect + 1 < idx.page_count() && idx.pages[expect + 1].start_row <= row) { ++expect; }
    REQUIRE(idx.find_page(row) == expect);
  }
  REQUIRE(idx.find_page(lengths.size()) == idx.page_count());  // out of range
}

TEST_CASE("row_byte_skiplist: cut_pages_with_skiplist reuses the baked directory",
          "[variable_width_page_index]")
{
  std::vector<std::size_t> lengths(8000, 37);
  constexpr std::size_t budget = 32 * 1024;
  auto const baked = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 8, budget,
                                             kVariableWidthOffsetsOverheadPerRow);
  auto const flat =
    cut_pages_with_skiplist(baked, budget, kVariableWidthOffsetsOverheadPerRow);
  REQUIRE(flat.size() == baked.page_count());
  for (std::size_t i = 0; i < flat.size(); ++i) { REQUIRE(flat[i] == baked.pages[i].start_row); }

  // A different budget than the baked one must still produce a correct cut.
  auto const other = cut_pages_with_skiplist(baked, budget / 4,
                                             kVariableWidthOffsetsOverheadPerRow);
  REQUIRE(other.size() > flat.size());
  REQUIRE(other.front() == 0);
  for (std::size_t i = 1; i < other.size(); ++i) { REQUIRE(other[i] > other[i - 1]); }
}

TEST_CASE("row_byte_skiplist: no page directory when no budget is given",
          "[variable_width_page_index]")
{
  std::vector<std::size_t> lengths(1000, 12);
  auto const idx = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 4);
  REQUIRE(idx.valid());
  REQUIRE(idx.page_count() == 0);
  REQUIRE(idx.page_size_bytes == 0);
  REQUIRE(idx.page_rows(0) == 0);
  REQUIRE(idx.page_bytes(0) == 0);
  REQUIRE(idx.find_page(0) == 0);
}

TEST_CASE("row_byte_skiplist: base_stride 1 is exact at every row", "[variable_width_page_index]")
{
  std::mt19937 rng{20260819};
  std::uniform_int_distribution<std::size_t> len{1, 200};
  std::vector<std::size_t> lengths(20000);
  for (auto& l : lengths) { l = len(rng); }

  // base 7 -- a non-zero first offset, as a sliced view would have.
  auto const idx = build_row_byte_skiplist(offsets_from_lengths(lengths, 7), 1, 8);
  REQUIRE(idx.valid());
  REQUIRE(idx.num_rows == lengths.size());
  REQUIRE(idx.level.size() > 1);  // levels were actually built

  std::int64_t exact = 0;
  for (std::size_t row = 0; row < lengths.size(); ++row) {
    REQUIRE(idx.bytes_before(row) == exact);  // exact, not interpolated
    exact += static_cast<std::int64_t>(lengths[row]);
  }
  REQUIRE(idx.total_bytes() == exact);
  REQUIRE(idx.bytes_before(lengths.size() + 99) == exact);  // clamped past the end
}

TEST_CASE("row_byte_skiplist: find_page_end matches an exhaustive scan",
          "[variable_width_page_index]")
{
  std::mt19937 rng{7};
  std::uniform_int_distribution<std::size_t> len{1, 60};
  std::vector<std::size_t> lengths(4000);
  for (auto& l : lengths) { l = len(rng); }
  auto const idx = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 4);
  REQUIRE(idx.valid());

  constexpr std::int64_t budget       = 1500;
  constexpr std::size_t per_row_cost  = 4;
  for (std::size_t start : {std::size_t{0}, std::size_t{1}, std::size_t{997},
                            std::size_t{2500}, lengths.size() - 2}) {
    // Ground truth: walk row by row until the budget is exceeded.
    std::int64_t acc  = 0;
    std::size_t expect = start;
    for (std::size_t r = start; r < lengths.size(); ++r) {
      acc += static_cast<std::int64_t>(lengths[r] + per_row_cost);
      if (acc > budget) { break; }
      expect = r + 1;
    }
    expect = std::max(expect, start + 1);
    REQUIRE(idx.find_page_end(start, budget, per_row_cost) == expect);
  }
}

TEST_CASE("cut_pages_with_skiplist: exact pages, every row covered once",
          "[variable_width_page_index]")
{
  std::mt19937 rng{99};
  std::uniform_int_distribution<std::size_t> len{1, 120};
  std::vector<std::size_t> lengths(30000);
  for (auto& l : lengths) { l = len(rng); }
  auto const idx = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 8);

  constexpr std::size_t budget       = 4096;
  constexpr std::size_t per_row_cost = kVariableWidthOffsetsOverheadPerRow;
  auto const starts = cut_pages_with_skiplist(idx, budget, per_row_cost);

  REQUIRE_FALSE(starts.empty());
  REQUIRE(starts.front() == 0);
  for (std::size_t i = 1; i < starts.size(); ++i) { REQUIRE(starts[i] > starts[i - 1]); }

  // Exact budgeting: with base_stride 1 there is no interpolation slack, so a
  // multi-row page must fit the budget outright.
  for (std::size_t i = 0; i < starts.size(); ++i) {
    auto const b = starts[i];
    auto const e = (i + 1 < starts.size()) ? starts[i + 1] : lengths.size();
    REQUIRE(e > b);
    std::int64_t cost = 0;
    for (std::size_t r = b; r < e; ++r) { cost += static_cast<std::int64_t>(lengths[r] + per_row_cost); }
    if (e - b > 1) { REQUIRE(cost <= static_cast<std::int64_t>(budget)); }
  }
  REQUIRE(starts.back() < lengths.size());
}

TEST_CASE("row_byte_skiplist: sparser base_stride still brackets the exact answer",
          "[variable_width_page_index]")
{
  std::vector<std::size_t> lengths(5000, 10);
  auto const dense  = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 4);
  auto const sparse = build_row_byte_skiplist(offsets_from_lengths(lengths), 16, 4);
  REQUIRE(dense.valid());
  REQUIRE(sparse.valid());
  // Uniform lengths: interpolation is exact even at stride 16.
  for (std::size_t row = 0; row <= lengths.size(); row += 137) {
    REQUIRE(sparse.bytes_before(row) == dense.bytes_before(row));
  }
  REQUIRE(sparse.footprint_bytes() < dense.footprint_bytes());
}

TEST_CASE("row_byte_skiplist: degenerate inputs", "[variable_width_page_index]")
{
  REQUIRE_FALSE(build_row_byte_skiplist({}, 1, 4).valid());
  REQUIRE_FALSE(build_row_byte_skiplist({5}, 1, 4).valid());       // no rows
  REQUIRE_FALSE(build_row_byte_skiplist({0, 10}, 0, 4).valid());   // zero stride
  REQUIRE_FALSE(build_row_byte_skiplist({0, 10}, 1, 1).valid());   // fanout must narrow
  auto const one = build_row_byte_skiplist({0, 10}, 1, 4);
  REQUIRE(one.valid());
  REQUIRE(one.num_rows == 1);
  REQUIRE(one.total_bytes() == 10);
  REQUIRE(cut_pages_with_skiplist(one, 4, 0) == std::vector<std::size_t>{0});
  REQUIRE(cut_pages_with_skiplist(one, 0, 0).empty());
}

TEST_CASE("row_byte_skiplist: a single oversized row cannot stall the cursor",
          "[variable_width_page_index]")
{
  std::vector<std::size_t> lengths{10, 10, 5000, 10, 10};
  auto const idx    = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 2);
  auto const starts = cut_pages_with_skiplist(idx, 100, 0);
  REQUIRE(starts.size() >= 3);
  REQUIRE(starts.back() < lengths.size());
  for (std::size_t i = 1; i < starts.size(); ++i) { REQUIRE(starts[i] > starts[i - 1]); }
}

/// Does cudf::column(sliced_strings_view) copy only the slice's characters, or
/// the whole parent chars buffer? The answer decides whether the exact-fit page
/// path silently carries the entire chunk in every page.
TEST_CASE("slice+copy of a strings column allocates only the slice",
          "[variable_width_page_index][gpu]")
{
  auto memory_manager = test_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = test_utils::get_resource_ref(*space);

  // 10k rows x 100 chars = ~1MB of characters; slice out 1% of the rows.
  std::vector<std::string> values;
  values.reserve(10000);
  for (int i = 0; i < 10000; ++i) { values.emplace_back(100, static_cast<char>('a' + i % 26)); }
  auto col = sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
    values, stream, mr);

  auto const whole_alloc = static_cast<std::size_t>(col->alloc_size());
  auto sliced = cudf::slice(col->view(), {0, 100});   // first 100 rows only
  REQUIRE_FALSE(sliced.empty());
  auto copied = std::make_shared<cudf::column>(sliced.front(), stream, mr);
  auto const slice_alloc = static_cast<std::size_t>(copied->alloc_size());

  WARN("whole column alloc_size=" << whole_alloc
       << "  1%-slice copy alloc_size=" << slice_alloc
       << "  ratio=" << (double)slice_alloc / (double)whole_alloc);

  CHECK(copied->size() == 100);
  // A slice holding 1% of the rows must not allocate anywhere near the whole
  // column; allow generous slack for the offsets child and alignment.
  CHECK(slice_alloc < whole_alloc / 4);
}

TEST_CASE("materialize_page_into_slab: fixed-size storage, exact contents",
          "[variable_width_page_index][gpu]")
{
  auto memory_manager = test_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = test_utils::get_resource_ref(*space);

  std::mt19937 rng{31337};
  std::uniform_int_distribution<int> len{1, 60};
  std::vector<std::string> values;
  values.reserve(4000);
  for (int i = 0; i < 4000; ++i) {
    values.emplace_back(static_cast<std::size_t>(len(rng)), static_cast<char>('a' + (i % 26)));
  }
  auto col = sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
    values, stream, mr);

  constexpr std::size_t kCharsSlab   = 8 * 1024;
  constexpr std::size_t kMaxRows     = 256;
  constexpr std::size_t kOffsetsSlab = (kMaxRows + 1) * sizeof(std::int32_t);

  std::vector<std::size_t> lengths;
  lengths.reserve(values.size());
  for (auto const& v : values) { lengths.push_back(v.size()); }
  auto const idx    = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 16);
  auto const starts = cut_pages_for_slabs(idx, kCharsSlab, kMaxRows);
  REQUIRE(starts.size() > 1);

  std::size_t covered = 0;
  std::size_t first_alloc = 0;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    auto const b = starts[i];
    auto const e = (i + 1 < starts.size()) ? starts[i + 1] : values.size();
    REQUIRE(e > b);
    REQUIRE(e - b <= kMaxRows);  // the offsets slab stays in bounds
    auto page =
      materialize_page_into_slab(col->view(), b, e - b, kCharsSlab, kOffsetsSlab, *space, stream);
    REQUIRE(page != nullptr);
    CHECK(page->size() == static_cast<cudf::size_type>(e - b));

    // Every page's device storage is the SAME size, however full it is -- the
    // whole point of the slab layout.
    auto const alloc = static_cast<std::size_t>(page->alloc_size());
    if (first_alloc == 0) { first_alloc = alloc; }
    CHECK(alloc == first_alloc);
    CHECK(alloc >= kCharsSlab + kOffsetsSlab);

    auto page_values = host_strings_from_column(page->view(), stream);
    REQUIRE(page_values.size() == e - b);
    for (std::size_t r = 0; r < page_values.size(); ++r) {
      CHECK(page_values[r] == values[b + r]);  // exact content, not just sizes
    }
    covered += e - b;
  }
  CHECK(covered == values.size());
}

TEST_CASE("cut_pages_for_slabs: respects both the chars budget and the row cap",
          "[variable_width_page_index]")
{
  std::vector<std::size_t> lengths(10000, 10);   // 10B rows
  auto const idx = build_row_byte_skiplist(offsets_from_lengths(lengths), 1, 8);

  // Row cap binds: 1000B of chars would allow 100 rows, cap allows 32.
  auto const capped = cut_pages_for_slabs(idx, 1000, 32);
  for (std::size_t i = 0; i + 1 < capped.size(); ++i) {
    REQUIRE(capped[i + 1] - capped[i] == 32);
  }
  // Chars budget binds: 200B allows 20 rows, cap allows 1000.
  auto const byte_bound = cut_pages_for_slabs(idx, 200, 1000);
  for (std::size_t i = 0; i + 1 < byte_bound.size(); ++i) {
    REQUIRE(byte_bound[i + 1] - byte_bound[i] == 20);
  }
  REQUIRE(cut_pages_for_slabs(idx, 0, 32).empty());
  REQUIRE(cut_pages_for_slabs(idx, 1000, 0).empty());
}

TEST_CASE("find_covering_variable_page: matches a naive linear scan for every row", "[variable_width_page_index]")
{
  std::mt19937 rng(42);
  std::uniform_int_distribution<std::size_t> len_dist(1, 4000);  // typical short-string byte lengths
  std::size_t const page_size_bytes = 16 * 1024;                 // 16KB, small on purpose to force many pages

  std::vector<std::size_t> lens(5000);
  for (auto& l : lens) { l = len_dist(rng); }
  // Sprinkle a few oversized rows in too.
  lens[10]   = page_size_bytes * 3;
  lens[2500] = page_size_bytes + 1;

  auto idx = index_variable_width_column_pages(lens, page_size_bytes);
  REQUIRE_FALSE(idx.pages.empty());
  CHECK(std::is_sorted(idx.page_start_rows.begin(), idx.page_start_rows.end()));

  for (std::size_t row = 0; row < lens.size(); row += 7) {  // sample every 7th row, full sweep is slow
    auto const* found    = find_covering_variable_page(idx, row);
    auto const expected_p = linear_scan_covering_page(lens, page_size_bytes, row);
    REQUIRE(found != nullptr);
    CHECK(found->page_index == expected_p);
    CHECK(row >= found->start_row);
    CHECK(row < found->start_row + found->num_rows);
  }
}

TEST_CASE("find_covering_variable_page: out-of-range and empty-index cases", "[variable_width_page_index]")
{
  std::vector<std::size_t> lens = {10, 10, 10};
  auto idx                      = index_variable_width_column_pages(lens, 100);
  CHECK(find_covering_variable_page(idx, 3) == nullptr);   // one past the chunk's last row
  CHECK(find_covering_variable_page(idx, 1000) == nullptr);

  variable_width_chunk_page_index empty_idx;
  CHECK(find_covering_variable_page(empty_idx, 0) == nullptr);
}

TEST_CASE("index_variable_width_column_pages: degenerate inputs", "[variable_width_page_index]")
{
  CHECK(index_variable_width_column_pages({}, 100).pages.empty());
  CHECK(index_variable_width_column_pages({10, 20}, 0).pages.empty());  // zero budget is not indexable

  // Every row individually oversized -- each gets its own one-row page.
  auto idx = index_variable_width_column_pages({500, 500, 500}, 100);
  REQUIRE(idx.pages.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(idx.pages[i].num_rows == 1);
    CHECK(idx.pages[i].start_row == i);
  }
}

TEST_CASE("index_variable_width_column_pages_from_view: real cudf STRING column, "
          "sliced page content matches the source rows exactly",
          "[variable_width_page_index][gpu]")
{
  auto memory_manager = test_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = test_utils::get_resource_ref(*space);

  // Mixed short/long values, page_size_bytes small enough to force several
  // pages and at least one page boundary landing mid-run of similar lengths.
  std::vector<std::string> values = {
    "alice", "bob", "carol_the_quick_brown_fox_jumps_over_the_lazy_dog", "dan", "eve",
    "frank", "grace_has_a_moderately_long_comment_field_here", "heidi", "ivan", "judy",
    "kate", "leo", "mallory_another_long_one_to_force_a_page_cut_right_here", "nina", "oscar",
  };
  auto col = sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
    values, stream, mr);

  std::size_t const page_size_bytes = 40;  // small on purpose: several pages over 15 rows
  auto idx = index_variable_width_column_pages_from_view(
    col->view(), page_size_bytes, /*chunk_index=*/3, "test_table", "test_col", *space, stream);

  REQUIRE_FALSE(idx.pages.empty());
  CHECK(idx.pages.size() > 1);  // budget is small enough that this must span multiple pages

  std::size_t covered_rows = 0;
  for (auto const& page : idx.pages) {
    CAPTURE(page.page_index, page.start_row, page.num_rows);
    REQUIRE(page.owned_column != nullptr);
    CHECK(page.table_name == "test_table");
    CHECK(page.column_name == "test_col");
    CHECK(page.chunk_index == 3);
    CHECK(page.memory_space == space);
    CHECK(page.state == variable_width_page_state::resident);
    CHECK(static_cast<std::size_t>(page.owned_column->size()) == page.num_rows);

    auto page_values = host_strings_from_column(page.owned_column->view(), stream);
    REQUIRE(page_values.size() == page.num_rows);
    for (std::size_t i = 0; i < page.num_rows; ++i) {
      CHECK(page_values[i] == values[page.start_row + i]);  // exact content, not just boundaries
    }
    covered_rows += page.num_rows;
  }
  CHECK(covered_rows == values.size());

  // Cross-check find_covering_variable_page against the same source data for
  // every row: the page it returns must actually contain that row's string.
  for (std::size_t row = 0; row < values.size(); ++row) {
    auto const* page = find_covering_variable_page(idx, row);
    REQUIRE(page != nullptr);
    auto page_values = host_strings_from_column(page->owned_column->view(), stream);
    CHECK(page_values[row - page->start_row] == values[row]);
  }
}

/// Hidden by the leading dot -- run explicitly with the tag. Compares the sparse
/// map against the skiplist at a real lineitem-chunk size, on build cost, cut
/// cost, host footprint and exactness.
TEST_CASE("skiplist vs sparse map at chunk scale", "[.skiplist_bench]")
{
  constexpr std::size_t kRows = 10'000'000;   // a lineitem chunk
  constexpr std::size_t kPageBytes = 16u << 20;
  std::mt19937 rng{2026};
  std::uniform_int_distribution<std::size_t> len{5, 120};  // l_comment-like spread

  std::vector<std::size_t> lengths(kRows);
  for (auto& l : lengths) { l = len(rng); }
  auto const offsets = offsets_from_lengths(lengths);

  auto time_ms = [](auto&& fn) {
    auto const t0 = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
      .count();
  };

  // Ground-truth cut, row by row.
  std::vector<std::size_t> truth;
  {
    std::size_t start = 0;
    while (start < kRows) {
      truth.push_back(start);
      std::int64_t acc = 0;
      std::size_t r = start;
      while (r < kRows) {
        acc += static_cast<std::int64_t>(lengths[r] + kVariableWidthOffsetsOverheadPerRow);
        if (acc > static_cast<std::int64_t>(kPageBytes)) { break; }
        ++r;
      }
      start = std::max(r, start + 1);
    }
  }

  for (std::size_t base : {std::size_t{1}, std::size_t{16}, std::size_t{256}}) {
    variable_width_row_byte_skiplist idx;
    auto const build = time_ms([&] { idx = build_row_byte_skiplist(offsets, base, 32); });
    std::vector<std::size_t> starts;
    auto const cut = time_ms([&] {
      starts = cut_pages_with_skiplist(idx, kPageBytes, kVariableWidthOffsetsOverheadPerRow);
    });
    std::size_t worst = 0;
    for (std::size_t i = 0; i < std::min(starts.size(), truth.size()); ++i) {
      worst = std::max(worst, starts[i] > truth[i] ? starts[i] - truth[i] : truth[i] - starts[i]);
    }
    WARN("skiplist base_stride=" << base << " levels=" << idx.level.size()
         << " footprint=" << idx.footprint_bytes() / (1024 * 1024) << "MB"
         << " build=" << build << "ms cut=" << cut << "ms"
         << " pages=" << starts.size() << " (truth " << truth.size() << ")"
         << " worst_boundary_error=" << worst << " rows");
  }

  // The sparse map currently in use, for reference.
  variable_width_offsets_extraction ex;
  ex.valid = true;
  ex.first_offset = offsets.front();
  ex.last_offset  = offsets.back();
  ex.sample_stride = (kRows + 2047) / 2048;
  for (std::size_t i = 0; i * ex.sample_stride <= kRows; ++i) {
    ex.sampled_offsets.push_back(offsets[i * ex.sample_stride]);
  }
  // col stays default-constructed; build_row_byte_map only reads col.size() for
  // num_rows, so drive the comparison off the same data via a direct build.
  variable_width_row_byte_map m;
  m.stride = ex.sample_stride;
  m.num_rows = kRows;
  for (auto s : ex.sampled_offsets) { m.byte_prefix.push_back(s - offsets.front()); }
  m.byte_prefix.back() = offsets.back() - offsets.front();
  std::vector<std::size_t> map_starts;
  auto const map_cut = time_ms([&] {
    map_starts = cut_pages_by_byte_budget(m, kPageBytes, kVariableWidthOffsetsOverheadPerRow);
  });
  std::size_t map_worst = 0;
  for (std::size_t i = 0; i < std::min(map_starts.size(), truth.size()); ++i) {
    map_worst = std::max(map_worst,
                         map_starts[i] > truth[i] ? map_starts[i] - truth[i]
                                                  : truth[i] - map_starts[i]);
  }
  WARN("sparse map (2048 samples) footprint=" << m.byte_prefix.size() * 8 / 1024 << "KB"
       << " cut=" << map_cut << "ms pages=" << map_starts.size()
       << " worst_boundary_error=" << map_worst << " rows");
}

TEST_CASE("choose_chars_slab_bytes: sizes the slab to the column's width",
          "[variable_width_page_index]")
{
  constexpr std::size_t kRows = 524288;
  constexpr std::size_t kMin  = 1u << 20;
  constexpr std::size_t kMax  = 16u << 20;

  // Utilization must stay high across the widths TPC-H actually presents,
  // instead of the 14-23% a single 16MiB slab gave narrow columns.
  struct { double bpr; char const* name; } const cases[] = {
    {1.0, "l_returnflag"}, {4.3, "l_shipmode"}, {12.0, "l_shipinstruct"}, {26.5, "l_comment"},
  };
  std::set<std::size_t> classes;
  for (auto const& c : cases) {
    auto const slab = choose_chars_slab_bytes(c.bpr, kRows, kMin, kMax);
    classes.insert(slab);
    auto const used = static_cast<double>(kRows) * c.bpr;
    CAPTURE(c.name, c.bpr, slab, used);
    CHECK(slab >= kMin);
    CHECK(slab <= kMax);
    // A power-of-two ladder wastes at most half a class -- except where the
    // min_bytes floor binds, which is deliberate: a sub-1MiB slab would make the
    // page count explode on the narrowest columns.
    if (slab > kMin && used <= static_cast<double>(kMax)) {
      CHECK(used > static_cast<double>(slab) / 2.0);
    }
    // Even at the floor, the page as a whole stays well utilized once the
    // offsets slab (which every column fills completely) is counted.
    auto const offsets_slab = (kRows + 1) * sizeof(std::int32_t);
    CHECK((used + static_cast<double>(offsets_slab)) /
            static_cast<double>(slab + offsets_slab) > 0.45);
  }
  // Still only a handful of distinct sizes, which is what keeps external
  // fragmentation near zero.
  CHECK(classes.size() <= 4);

  // Degenerate inputs fall back to the maximum rather than returning 0.
  CHECK(choose_chars_slab_bytes(10.0, 0, kMin, kMax) == kMax);
  CHECK(choose_chars_slab_bytes(10.0, kRows, 0, kMax) == kMax);
  // A very wide column is clamped, not grown without bound.
  CHECK(choose_chars_slab_bytes(1e6, kRows, kMin, kMax) == kMax);
}

TEST_CASE("find_page_starts_on_device: matches a row-by-row cut, no host index",
          "[variable_width_page_index][gpu]")
{
  auto memory_manager = test_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = cudf::get_default_stream();
  auto mr     = test_utils::get_resource_ref(*space);

  std::mt19937 rng{99991};
  std::uniform_int_distribution<int> len{1, 90};
  std::vector<std::string> values;
  values.reserve(30000);
  for (int i = 0; i < 30000; ++i) {
    values.emplace_back(static_cast<std::size_t>(len(rng)), static_cast<char>('a' + i % 26));
  }
  auto col = sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
    values, stream, mr);

  constexpr std::size_t kBudget  = 16 * 1024;
  constexpr std::size_t kMaxRows = 100000;   // row cap slack: the byte grid binds
  auto const starts = find_page_starts_on_device(col->view(), kBudget, kMaxRows, stream);

  REQUIRE_FALSE(starts.empty());
  CHECK(starts.front() == 0);
  for (std::size_t i = 1; i < starts.size(); ++i) { CHECK(starts[i] > starts[i - 1]); }
  CHECK(starts.back() < values.size());

  // Ground truth: walk rows accumulating bytes. The device search targets an even
  // byte grid, so a page may overshoot by at most one row -- assert that bound
  // rather than exact equality with the greedy walk.
  std::size_t worst_over = 0;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    auto const b = starts[i];
    auto const e = (i + 1 < starts.size()) ? starts[i + 1] : values.size();
    std::size_t bytes = 0, longest = 0;
    for (std::size_t r = b; r < e; ++r) {
      bytes += values[r].size();
      longest = std::max(longest, values[r].size());
    }
    CAPTURE(i, b, e, bytes);
    if (e - b > 1) { CHECK(bytes <= kBudget + longest); }
    worst_over = std::max(worst_over, bytes > kBudget ? bytes - kBudget : 0);
  }
  WARN("pages=" << starts.size() << " worst overshoot=" << worst_over << " bytes (max row 90)");

  // The row cap binds instead when it is the tighter constraint.
  auto const capped = find_page_starts_on_device(col->view(), kBudget, 64, stream);
  REQUIRE(capped.size() > 1);
  for (std::size_t i = 1; i < capped.size(); ++i) { CHECK(capped[i] - capped[i - 1] <= 64); }

  // Degenerate inputs return nothing rather than misbehaving.
  CHECK(find_page_starts_on_device(col->view(), 0, 64, stream).empty());
  CHECK(find_page_starts_on_device(col->view(), kBudget, 0, stream).empty());
}
