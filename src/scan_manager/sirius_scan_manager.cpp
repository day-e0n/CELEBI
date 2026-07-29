/*
 * Copyright 2025, Sirius Contributors.
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

#include "scan_manager/sirius_scan_manager.hpp"

#include "data/data_batch_utils.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"
#include "io/parquet_helpers.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_metadata.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "planner/query.hpp"
#include "scan_manager/round_robin_strategy.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/reduction.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <atomic>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace sirius::scan_manager {

namespace {

struct cached_databatch_provider : public databatch_provider {
  explicit cached_databatch_provider(pinned_entry const& entry, std::span<size_t> selected_columns)
    : _entry(entry)
  {
    auto const& entry_column_names = _entry.cache_info.column_names();
    std::ranges::for_each(selected_columns, [this, &entry_column_names](size_t idx) {
      _column_names.emplace_back(entry_column_names[idx]);
      _column_indices.push_back(idx);
    });

    if (_entry.tier == cucascade::memory::Tier::GPU) {
      if (_entry.data_batches_by_column.empty()) {
        _n_chunks = 0;
      } else {
        _n_chunks = _entry.data_batches_by_column.begin()->second.size();
      }
    } else if (_entry.tier == cucascade::memory::Tier::HOST) {
      _n_chunks = _entry.host_chunks.size();
    }

    // Resolve every chunk now, while the caller (try_assign_cached_entries) still holds
    // _pinned_entries_mutex -- see the matching comment on fixed_page_databatch_provider.
    // get_next_batch() runs later, unlocked, on pipeline worker threads; reading _entry's
    // vectors lazily there could race a concurrent insert_fixed_page_entry_from_view()
    // append.
    _prebuilt_batches.reserve(_n_chunks);
    for (std::size_t index = 0; index < _n_chunks; ++index) {
      if (_entry.tier == cucascade::memory::Tier::GPU) {
        _prebuilt_batches.push_back(get_device_databatch(index));
      } else if (_entry.tier == cucascade::memory::Tier::HOST) {
        _prebuilt_batches.push_back(get_host_databatch(index));
      }
    }
  }

  std::shared_ptr<cucascade::data_batch> get_next_batch() override
  {
    auto index = _index.fetch_add(1);
    if (index >= _prebuilt_batches.size()) { return nullptr; }
    return _prebuilt_batches[index];
  }

 private:
  std::shared_ptr<cucascade::data_batch> get_host_databatch(std::size_t index)
  {
    if (index >= _entry.host_chunks.size()) { return nullptr; }
    const auto& chunk = _entry.host_chunks.at(index);
    if (!chunk) { return nullptr; }
    auto data_rep = chunk->slice(_column_indices);
    return cucascade::data_batch::make(get_next_batch_id(), std::move(data_rep));
  }

  std::shared_ptr<cucascade::data_batch> get_device_databatch(std::size_t index)
  {
    if (index >= _entry.chunk_memory_spaces.size()) { return nullptr; }
    std::vector<std::shared_ptr<cudf::column>> columns;
    std::vector<cudf::column_view> column_views;
    std::size_t alloc_size = 0;
    for (const auto& col_idx : _column_names) {
      const auto& col_chunks = _entry.data_batches_by_column.at(col_idx);
      if (index >= col_chunks.size()) { return nullptr; }
      columns.push_back(col_chunks.at(index));
      column_views.emplace_back(columns.back()->view());
      alloc_size += columns.back()->alloc_size();
    }
    cudf::table_view view(column_views);
    auto* chunk_space = !_entry.chunk_memory_spaces.empty() ? _entry.chunk_memory_spaces.at(index)
                                                            : _entry.memory_space;
    auto gpu_repr     = std::make_unique<::cucascade::gpu_table_representation>(
      view, std::move(columns), alloc_size, *chunk_space, rmm::cuda_stream_view{});
    return ::cucascade::data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));
  }

  std::size_t _n_chunks;
  std::vector<std::string> _column_names;
  std::vector<size_t> _column_indices;
  const pinned_entry& _entry;
  std::atomic<std::size_t> _index{0};
  std::vector<std::shared_ptr<cucascade::data_batch>> _prebuilt_batches;
};

struct fixed_page_batch_range {
  std::size_t chunk_index{0};
  std::size_t row_offset{0};
  std::size_t num_rows{0};
  std::size_t page_count{1};
  cucascade::memory::memory_space* memory_space{nullptr};
};

bool fixed_page_backed_provider_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_BACKED_PROVIDER");
  return value != nullptr && std::string_view(value) == "1";
}

bool fixed_page_hybrid_provider_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_HYBRID_PROVIDER");
  return value != nullptr && std::string_view(value) == "1";
}

std::size_t fixed_page_provider_coalesce_pages()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_PROVIDER_COALESCE_PAGES");
  if (value == nullptr || std::string_view(value).empty()) { return 1; }
  char* end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_PROVIDER_COALESCE_PAGES='{}'; using 1",
      value);
    return 1;
  }
  return std::min<std::size_t>(static_cast<std::size_t>(parsed), 1024);
}

std::uint64_t next_fixed_width_page_lru_tick()
{
  static std::atomic<std::uint64_t> tick{1};
  return tick.fetch_add(1, std::memory_order_relaxed) + 1;
}

void touch_fixed_width_page(fixed_width_column_page const& page)
{
  page.last_access_tick.store(next_fixed_width_page_lru_tick(), std::memory_order_relaxed);
}

bool fixed_width_page_is_active(fixed_width_column_page const& page)
{
  return page.active_reader_count.load(std::memory_order_relaxed) != 0;
}

void adjust_fixed_width_page_active_reader(fixed_width_column_page const& page, int delta)
{
  if (delta > 0) {
    page.active_reader_count.fetch_add(static_cast<std::uint32_t>(delta),
                                       std::memory_order_relaxed);
    return;
  }
  auto const decrement = static_cast<std::uint32_t>(-delta);
  auto current         = page.active_reader_count.load(std::memory_order_relaxed);
  while (current != 0) {
    auto const next = current > decrement ? current - decrement : 0;
    if (page.active_reader_count.compare_exchange_weak(current,
                                                       next,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
      return;
    }
  }
}

bool fixed_page_covers_range(fixed_width_column_page const& page,
                             std::size_t chunk_index,
                             std::size_t row_offset,
                             std::size_t num_rows)
{
  if (page.state != fixed_width_page_state::resident || page.chunk_index != chunk_index) {
    return false;
  }
  auto const page_begin = page.row_offset;
  auto const page_end   = page.row_offset + page.num_rows;
  auto const range_end  = row_offset + num_rows;
  return page_begin <= row_offset && range_end <= page_end;
}

fixed_width_column_page const* find_covering_fixed_page(
  pinned_entry const& entry,
  std::string const& column_name,
  std::size_t chunk_index,
  std::size_t row_offset,
  std::size_t num_rows)
{
  auto pages_it = entry.fixed_width_pages_by_column.find(column_name);
  if (pages_it == entry.fixed_width_pages_by_column.end()) { return nullptr; }
  for (auto const& page : pages_it->second) {
    if (fixed_page_covers_range(page, chunk_index, row_offset, num_rows)) { return &page; }
  }
  return nullptr;
}

class fixed_page_databatch_provider final : public databatch_provider {
 public:
  explicit fixed_page_databatch_provider(pinned_entry const& entry,
                                         std::span<size_t> selected_columns)
    : _entry(entry)
  {
    auto const& entry_column_names = _entry.cache_info.column_names();
    std::ranges::for_each(selected_columns, [this, &entry_column_names](size_t idx) {
      _column_names.emplace_back(entry_column_names[idx]);
    });
    build_ranges();
    adjust_active_pages(1);
    // Materialize every batch now, while the caller (try_assign_cached_entries) still
    // holds _pinned_entries_mutex. get_next_batch() is invoked later from pipeline
    // worker threads with no lock held; if it read _entry's containers lazily at that
    // point, a concurrent insert_fixed_page_entry_from_view() append (which mutates
    // those same std::vectors under the mutex) could race a vector reallocation
    // against this read -- undefined behavior that surfaced as intermittent, exact
    // duplicate rows. Resolving everything up front removes that read from the
    // unsynchronized path entirely.
    _prebuilt_batches.reserve(_ranges.size());
    for (auto const& range : _ranges) {
      _prebuilt_batches.push_back(build_device_databatch(range));
    }
  }

  ~fixed_page_databatch_provider() override { adjust_active_pages(-1); }

  [[nodiscard]] bool usable() const noexcept
  {
    return !_ranges.empty() && _covered_rows == _entry.num_rows;
  }

  [[nodiscard]] std::size_t batch_count() const noexcept { return _ranges.size(); }

  std::shared_ptr<cucascade::data_batch> get_next_batch() override
  {
    auto index = _index.fetch_add(1);
    if (index >= _prebuilt_batches.size()) { return nullptr; }
    return _prebuilt_batches[index];
  }

 private:
  std::shared_ptr<cucascade::data_batch> build_device_databatch(fixed_page_batch_range const& range)
  {
    if (!range.memory_space) { return nullptr; }

    if (range.memory_space->get_device_id() >= 0) {
      rmm::cuda_set_device_raii device_guard{
        rmm::cuda_device_id{range.memory_space->get_device_id()}};
      return get_device_databatch(range);
    }
    return get_device_databatch(range);
  }

  void adjust_active_pages(int delta) const
  {
    if (_ranges.empty() || delta == 0) { return; }
    for (auto const& range : _ranges) {
      auto const range_end = range.row_offset + range.num_rows;
      for (auto const& column_name : _column_names) {
        auto cursor = range.row_offset;
        while (cursor < range_end) {
          auto const* page = find_covering_fixed_page(_entry, column_name, range.chunk_index, cursor, 1);
          if (!page || !page->owned_column) { break; }
          adjust_fixed_width_page_active_reader(*page, delta);
          auto const page_end = page->row_offset + page->num_rows;
          if (page_end <= cursor) { break; }
          cursor = std::min(page_end, range_end);
        }
      }
    }
  }

  void build_ranges()
  {
    if (_entry.tier != cucascade::memory::Tier::GPU || _column_names.empty() ||
        _entry.fixed_width_pages_by_column.empty()) {
      return;
    }

    auto driver = choose_driver_column();
    if (!driver.has_value()) { return; }
    auto const& driver_pages = _entry.fixed_width_pages_by_column.at(_column_names[*driver]);
    auto const max_coalesce_pages = fixed_page_provider_coalesce_pages();
    for (std::size_t i = 0; i < driver_pages.size(); ++i) {
      auto const& page = driver_pages[i];
      if (page.state != fixed_width_page_state::resident || page.num_rows == 0) { continue; }
      if (!all_columns_cover_tiled(page.chunk_index, page.row_offset, page.num_rows)) { continue; }

      std::size_t coalesced_rows  = page.num_rows;
      std::size_t coalesced_pages = 1;
      auto const range_begin      = page.row_offset;
      while (coalesced_pages < max_coalesce_pages && i + coalesced_pages < driver_pages.size()) {
        auto const& next = driver_pages[i + coalesced_pages];
        if (next.state != fixed_width_page_state::resident || next.num_rows == 0) { break; }
        if (next.chunk_index != page.chunk_index || next.memory_space != page.memory_space) { break; }
        if (next.row_offset != range_begin + coalesced_rows) { break; }
        auto const next_rows = coalesced_rows + next.num_rows;
        if (!all_columns_cover_tiled(page.chunk_index, range_begin, next_rows)) { break; }
        coalesced_rows = next_rows;
        ++coalesced_pages;
      }

      _ranges.push_back(fixed_page_batch_range{page.chunk_index,
                                               range_begin,
                                               coalesced_rows,
                                               coalesced_pages,
                                               page.memory_space});
      _covered_rows += coalesced_rows;
      i += coalesced_pages - 1;
    }
  }

  [[nodiscard]] std::optional<std::size_t> choose_driver_column() const
  {
    std::optional<std::size_t> selected;
    std::size_t selected_rows = std::numeric_limits<std::size_t>::max();
    bool const hybrid = fixed_page_hybrid_provider_enabled();
    for (std::size_t i = 0; i < _column_names.size(); ++i) {
      auto pages_it = _entry.fixed_width_pages_by_column.find(_column_names[i]);
      if (pages_it == _entry.fixed_width_pages_by_column.end() || pages_it->second.empty()) {
        if (hybrid && has_chunk_column(_column_names[i])) { continue; }
        return std::nullopt;
      }
      std::size_t min_rows = std::numeric_limits<std::size_t>::max();
      bool has_resident    = false;
      for (auto const& page : pages_it->second) {
        if (page.state != fixed_width_page_state::resident || page.num_rows == 0) { continue; }
        has_resident = true;
        min_rows     = std::min(min_rows, page.num_rows);
      }
      if (!has_resident) { return std::nullopt; }
      if (min_rows < selected_rows) {
        selected      = i;
        selected_rows = min_rows;
      }
    }
    return selected;
  }

  [[nodiscard]] bool column_pages_cover_tiled(std::string const& column_name,
                                              std::size_t chunk_index,
                                              std::size_t row_offset,
                                              std::size_t num_rows) const
  {
    auto const range_end = row_offset + num_rows;
    auto cursor         = row_offset;
    while (cursor < range_end) {
      auto const* page = find_covering_fixed_page(_entry, column_name, chunk_index, cursor, 1);
      if (!page || !page->owned_column) { return false; }
      auto const page_end = page->row_offset + page->num_rows;
      if (page_end <= cursor) { return false; }
      cursor = std::min(page_end, range_end);
    }
    return true;
  }

  [[nodiscard]] bool all_columns_cover_tiled(std::size_t chunk_index,
                                             std::size_t row_offset,
                                             std::size_t num_rows) const
  {
    for (auto const& column_name : _column_names) {
      if (find_covering_fixed_page(_entry, column_name, chunk_index, row_offset, num_rows)) {
        continue;
      }
      if (column_pages_cover_tiled(column_name, chunk_index, row_offset, num_rows)) {
        continue;
      }
      if (!fixed_page_hybrid_provider_enabled()) { return false; }
      if (!chunk_column_covers(column_name, chunk_index, row_offset, num_rows)) { return false; }
    }
    return true;
  }

  [[nodiscard]] bool has_chunk_column(std::string const& column_name) const
  {
    auto chunks_it = _entry.data_batches_by_column.find(column_name);
    if (chunks_it == _entry.data_batches_by_column.end() || chunks_it->second.empty()) {
      return false;
    }
    for (auto const& chunk : chunks_it->second) {
      if (!chunk) { return false; }
    }
    return true;
  }

  [[nodiscard]] bool chunk_column_covers(std::string const& column_name,
                                         std::size_t chunk_index,
                                         std::size_t row_offset,
                                         std::size_t num_rows) const
  {
    auto chunks_it = _entry.data_batches_by_column.find(column_name);
    if (chunks_it == _entry.data_batches_by_column.end() ||
        chunk_index >= chunks_it->second.size()) {
      return false;
    }
    auto const& chunk = chunks_it->second.at(chunk_index);
    if (!chunk) { return false; }
    return row_offset + num_rows <= static_cast<std::size_t>(chunk->size());
  }

  std::shared_ptr<cudf::column> materialize_owned_subrange(
    fixed_width_column_page const& page,
    std::size_t row_offset,
    std::size_t num_rows) const
  {
    if (!page.owned_column) { return nullptr; }
    if (row_offset == page.row_offset && num_rows == page.num_rows) { return page.owned_column; }
    auto const begin = static_cast<cudf::size_type>(row_offset - page.row_offset);
    auto const end   = static_cast<cudf::size_type>(begin + num_rows);
    auto views       = cudf::slice(page.owned_column->view(), {begin, end});
    if (views.empty()) { return nullptr; }
    return std::make_shared<cudf::column>(
      views.front(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  }

  std::shared_ptr<cudf::column> materialize_chunk_subrange(
    std::string const& column_name,
    fixed_page_batch_range const& range) const
  {
    auto chunks_it = _entry.data_batches_by_column.find(column_name);
    if (chunks_it == _entry.data_batches_by_column.end() ||
        range.chunk_index >= chunks_it->second.size()) {
      return nullptr;
    }
    auto column = chunks_it->second.at(range.chunk_index);
    if (!column) { return nullptr; }
    auto views = cudf::slice(column->view(),
                             {static_cast<cudf::size_type>(range.row_offset),
                              static_cast<cudf::size_type>(range.row_offset + range.num_rows)});
    if (views.empty()) { return nullptr; }
    return std::make_shared<cudf::column>(
      views.front(), cudf::get_default_stream(), range.memory_space->get_default_allocator());
  }

  std::shared_ptr<cudf::column> materialize_page_tiled_column(
    std::string const& column_name,
    fixed_page_batch_range const& range) const
  {
    auto const range_end = range.row_offset + range.num_rows;
    auto cursor         = range.row_offset;
    std::vector<std::shared_ptr<cudf::column>> pieces;
    std::vector<cudf::column_view> piece_views;
    pieces.reserve(range.page_count);
    piece_views.reserve(range.page_count);

    while (cursor < range_end) {
      auto const* page = find_covering_fixed_page(_entry, column_name, range.chunk_index, cursor, 1);
      if (!page || !page->owned_column) { return nullptr; }
      touch_fixed_width_page(*page);
      auto const page_end  = page->row_offset + page->num_rows;
      auto const piece_end = std::min(page_end, range_end);
      if (piece_end <= cursor) { return nullptr; }
      auto owned = materialize_owned_subrange(*page, cursor, piece_end - cursor);
      if (!owned) { return nullptr; }
      piece_views.emplace_back(owned->view());
      pieces.push_back(std::move(owned));
      cursor = piece_end;
    }

    if (pieces.empty()) { return nullptr; }
    if (pieces.size() == 1) { return pieces.front(); }
    auto concatenated = cudf::concatenate(piece_views, cudf::get_default_stream(),
                                          range.memory_space->get_default_allocator());
    return std::shared_ptr<cudf::column>{std::move(concatenated)};
  }

  std::shared_ptr<cudf::column> materialize_column_range(
    std::string const& column_name,
    fixed_page_batch_range const& range) const
  {
    auto const* page = find_covering_fixed_page(_entry, column_name, range.chunk_index,
                                                range.row_offset, range.num_rows);
    if (page) {
      touch_fixed_width_page(*page);
      if (auto owned = materialize_owned_subrange(*page, range.row_offset, range.num_rows)) {
        return owned;
      }
    }

    if (auto tiled = materialize_page_tiled_column(column_name, range)) { return tiled; }

    if (!fixed_page_hybrid_provider_enabled()) { return nullptr; }
    return materialize_chunk_subrange(column_name, range);
  }

  std::shared_ptr<cucascade::data_batch> get_device_databatch(fixed_page_batch_range const& range)
  {
    std::vector<std::shared_ptr<cudf::column>> columns;
    std::vector<cudf::column_view> column_views;
    std::size_t alloc_size = 0;
    columns.reserve(_column_names.size());
    column_views.reserve(_column_names.size());

    for (auto const& column_name : _column_names) {
      auto column = materialize_column_range(column_name, range);
      if (!column) { return nullptr; }
      column_views.emplace_back(column->view());
      alloc_size += column->alloc_size();
      columns.push_back(std::move(column));
    }

    cudf::table_view view(column_views);
    auto gpu_repr = std::make_unique<::cucascade::gpu_table_representation>(
      view, std::move(columns), alloc_size, *range.memory_space, rmm::cuda_stream_view{});
    return ::cucascade::data_batch::make(::sirius::get_next_batch_id(), std::move(gpu_repr));
  }

  std::vector<std::string> _column_names;
  std::vector<fixed_page_batch_range> _ranges;
  std::size_t _covered_rows{0};
  pinned_entry const& _entry;
  std::atomic<std::size_t> _index{0};
  std::vector<std::shared_ptr<cucascade::data_batch>> _prebuilt_batches;
};

bool has_chunk_backing_for_selected_columns(pinned_entry const& entry,
                                            std::span<size_t> selected_columns)
{
  if (entry.tier != cucascade::memory::Tier::GPU) { return true; }
  auto const& names = entry.cache_info.column_names();
  for (auto const idx : selected_columns) {
    if (idx >= names.size()) { return false; }
    auto it = entry.data_batches_by_column.find(names[idx]);
    if (it == entry.data_batches_by_column.end() || it->second.empty()) { return false; }
    for (auto const& chunk : it->second) {
      if (!chunk) { return false; }
    }
  }
  return true;
}

std::unique_ptr<databatch_provider> make_provider_for_pinned_entry(
  pinned_entry const& entry, std::span<size_t> selected_columns)
{
  if (fixed_page_backed_provider_enabled()) {
    auto fixed_page_provider =
      std::make_unique<fixed_page_databatch_provider>(entry, selected_columns);
    if (fixed_page_provider->usable()) {
      SIRIUS_LOG_INFO("[fixed-page-cache] using {} cached provider batches={}",
                      fixed_page_hybrid_provider_enabled() ? "hybrid page/chunk" : "page-backed",
                      fixed_page_provider->batch_count());
      return fixed_page_provider;
    }
    SIRIUS_LOG_INFO("[fixed-page-cache] page-backed cached provider unavailable; falling back to "
                    "chunk-backed cached provider");
  }
  if (!has_chunk_backing_for_selected_columns(entry, selected_columns)) {
    SIRIUS_LOG_INFO("[fixed-page-cache] cached provider unavailable; selected columns have "
                    "page-only storage with incomplete resident coverage");
    return nullptr;
  }
  return std::make_unique<cached_databatch_provider>(entry, selected_columns);
}

/// Strip a leading "file://" scheme (case-insensitive) so the path can be
/// resolved by a local-file backend.
std::string normalize_path(std::string const& p)
{
  static constexpr std::string_view kFile = "file://";
  if (p.size() > kFile.size()) {
    bool is_file_uri = true;
    for (std::size_t i = 0; i < kFile.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(p[i])) != static_cast<unsigned char>(kFile[i])) {
        is_file_uri = false;
        break;
      }
    }
    if (is_file_uri) { return p.substr(kFile.size()); }
  }
  return p;
}

// wdy start
std::size_t fixed_width_page_size_bytes()
{
  static constexpr std::size_t kDefaultPageBytes = 16ULL * 1024ULL * 1024ULL;
  auto const* value                              = std::getenv("SIRIUS_FIXED_WIDTH_PAGE_BYTES");
  if (value == nullptr || value[0] == '\0') { return kDefaultPageBytes; }
  try {
    auto parsed = static_cast<std::size_t>(std::stoull(value));
    return parsed == 0 ? kDefaultPageBytes : parsed;
  } catch (...) {
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] invalid SIRIUS_FIXED_WIDTH_PAGE_BYTES='{}'; using default {}",
      value,
      kDefaultPageBytes);
    return kDefaultPageBytes;
  }
}


std::size_t parse_byte_size_or_zero(char const* value)
{
  if (value == nullptr || value[0] == '\0') { return 0; }
  try {
    std::string text{value};
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
      text.pop_back();
    }
    std::size_t pos = 0;
    auto parsed     = std::stoull(text, &pos);
    auto suffix     = text.substr(pos);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    if (suffix == "gb" || suffix == "gib" || suffix == "g") {
      parsed *= 1024ULL * 1024ULL * 1024ULL;
    } else if (suffix == "mb" || suffix == "mib" || suffix == "m") {
      parsed *= 1024ULL * 1024ULL;
    } else if (suffix == "kb" || suffix == "kib" || suffix == "k") {
      parsed *= 1024ULL;
    } else if (!suffix.empty() && suffix != "b") {
      return 0;
    }
    return static_cast<std::size_t>(parsed);
  } catch (...) {
    return 0;
  }
}

std::size_t fixed_page_cache_budget_bytes_per_gpu()
{
  auto const budget = parse_byte_size_or_zero(std::getenv("SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"));
  if (budget == 0 && std::getenv("SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU") != nullptr) {
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU='{}'; budget disabled",
      std::getenv("SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"));
  }
  if (budget == 0) { return 0; }

  auto reserve =
    parse_byte_size_or_zero(std::getenv("SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU"));
  if (reserve == 0 &&
      std::getenv("SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU") != nullptr) {
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU='{}'; "
      "workspace reserve disabled",
      std::getenv("SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU"));
  }
  if (reserve >= budget) {
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] workspace reserve {} is >= page budget {}; page budget disabled",
      reserve,
      budget);
    return 0;
  }
  auto const effective_budget = budget - reserve;
  static std::atomic<bool> logged_budget{false};
  bool expected = false;
  if (logged_budget.compare_exchange_strong(expected, true)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] budget configured_bytes_per_gpu={} workspace_reserve_bytes_per_gpu={} "
      "resident_page_budget_bytes_per_gpu={}",
      budget,
      reserve,
      effective_budget);
  }
  return effective_budget;
}


std::size_t fixed_page_cache_min_free_bytes_per_gpu()
{
  auto const* configured_text = std::getenv("SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU");
  if (configured_text == nullptr || configured_text[0] == '\0') { return 0; }

  auto const min_free = parse_byte_size_or_zero(configured_text);
  if (min_free == 0 && std::string_view(configured_text) != "0") {
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU='{}'; "
      "memory-pressure eviction disabled",
      configured_text);
    return 0;
  }

  static std::atomic<bool> logged_min_free{false};
  bool expected = false;
  if (min_free != 0 && logged_min_free.compare_exchange_strong(expected, true)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] memory-pressure eviction configured min_free_bytes_per_gpu={}",
      min_free);
  }
  return min_free;
}

std::size_t fixed_page_cache_evict_headroom_bytes_per_gpu(std::size_t budget)
{
  if (budget == 0) { return 0; }

  auto const* configured_text =
    std::getenv("SIRIUS_FIXED_PAGE_CACHE_EVICT_HEADROOM_BYTES_PER_GPU");
  std::size_t headroom = 512ULL * 1024ULL * 1024ULL;
  if (configured_text != nullptr && configured_text[0] != '\0') {
    headroom = parse_byte_size_or_zero(configured_text);
    if (headroom == 0 && std::string_view(configured_text) != "0") {
      SIRIUS_LOG_WARN(
        "[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_CACHE_EVICT_HEADROOM_BYTES_PER_GPU='{}'; "
        "using default 512MiB",
        configured_text);
      headroom = 512ULL * 1024ULL * 1024ULL;
    }
  }

  headroom = std::min(headroom, budget / 2);
  static std::atomic<bool> logged_headroom{false};
  bool expected = false;
  if (logged_headroom.compare_exchange_strong(expected, true)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] eviction headroom bytes_per_gpu={} low_watermark_bytes_per_gpu={} ",
      headroom,
      budget > headroom ? budget - headroom : budget);
  }
  return headroom;
}

std::size_t fixed_page_cache_low_watermark_bytes_per_gpu(std::size_t budget)
{
  auto const headroom = fixed_page_cache_evict_headroom_bytes_per_gpu(budget);
  return budget > headroom ? budget - headroom : budget;
}

bool is_fixed_width_page_candidate(cudf::column_view const& col) noexcept;

std::size_t fixed_page_admission_max_entry_bytes()
{
  auto const* configured_text = std::getenv("SIRIUS_FIXED_PAGE_ADMISSION_MAX_ENTRY_BYTES");
  if (configured_text != nullptr) {
    auto const configured = parse_byte_size_or_zero(configured_text);
    if (configured == 0 && configured_text[0] != '\0' && std::string_view(configured_text) != "0") {
      SIRIUS_LOG_WARN(
        "[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_ADMISSION_MAX_ENTRY_BYTES='{}'; "
        "size-aware admission disabled",
        configured_text);
    }
    return configured;
  }

  auto const budget = fixed_page_cache_budget_bytes_per_gpu();
  if (budget == 0) { return 0; }
  auto const admission_limit = budget / 2;
  static std::atomic<bool> logged_admission{false};
  bool expected = false;
  if (logged_admission.compare_exchange_strong(expected, true)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] admission max_entry_bytes={} source=budget_half",
      admission_limit);
  }
  return admission_limit;
}

std::size_t fixed_width_table_view_bytes(cudf::table_view view)
{
  std::size_t bytes = 0;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    auto const col = view.column(i);
    if (col.size() <= 0) { continue; }
    if (is_fixed_width_page_candidate(col)) {
      auto const element_size = cudf::size_of(col.type());
      if (element_size == 0) { continue; }
      bytes += static_cast<std::size_t>(col.size()) * element_size;
      continue;
    }
    // Conservative admission estimate for variable-width hybrid chunks.
    bytes += static_cast<std::size_t>(col.size()) * 32ULL;
  }
  return bytes;
}

std::size_t fixed_width_entry_logical_bytes(pinned_entry const& entry)
{
  std::size_t bytes = 0;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : col_pages) {
      bytes += page.num_bytes;
    }
  }
  for (auto const& [_, chunks] : entry.data_batches_by_column) {
    for (auto const& chunk : chunks) {
      if (chunk) { bytes += chunk->alloc_size(); }
    }
  }
  return bytes;
}

bool fixed_page_owned_pages_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_OWNED_PAGES");
  return value != nullptr && std::string_view(value) == "1";
}

bool is_auto_fixed_page_entry(std::string const& name)
{
  return name.rfind("__wdy_auto_fixed_page:", 0) == 0;
}

bool is_fixed_width_page_candidate(cudf::column_view const& col) noexcept
{
  switch (col.type().id()) {
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::EMPTY: return false;
    default: return true;
  }
}

bool fixed_page_name_contains(std::string const& text, std::string_view needle)
{
  auto it = std::search(text.begin(),
                        text.end(),
                        needle.begin(),
                        needle.end(),
                        [](unsigned char a, unsigned char b) {
                          return std::tolower(a) == std::tolower(b);
                        });
  return it != text.end();
}

double fixed_width_page_admission_score(std::string const& column_name,
                                        cudf::type_id type_id,
                                        fixed_width_page_stats const& stats)
{
  double score = 100.0;
  if (stats.valid) { score += 50.0; }
  if (stats.has_null) { score -= 15.0; }

  switch (type_id) {
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: score += 300.0; break;
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64: score += 40.0; break;
    default: break;
  }

  if (fixed_page_name_contains(column_name, "date") ||
      fixed_page_name_contains(column_name, "time")) {
    score += 350.0;
  }
  if (fixed_page_name_contains(column_name, "key") ||
      fixed_page_name_contains(column_name, "id")) {
    score += 250.0;
  }
  if (fixed_page_name_contains(column_name, "shipdate") ||
      fixed_page_name_contains(column_name, "orderdate") ||
      fixed_page_name_contains(column_name, "receiptdate") ||
      fixed_page_name_contains(column_name, "commitdate")) {
    score += 250.0;
  }
  if (fixed_page_name_contains(column_name, "comment") ||
      fixed_page_name_contains(column_name, "name") ||
      fixed_page_name_contains(column_name, "address")) {
    score -= 75.0;
  }
  return score;
}

template <typename T>
fixed_width_page_stats make_signed_page_stats(cudf::scalar const& min_scalar,
                                              cudf::scalar const& max_scalar,
                                              bool has_null,
                                              rmm::cuda_stream_view stream)
{
  fixed_width_page_stats stats;
  stats.valid      = true;
  stats.has_null   = has_null;
  stats.kind       = fixed_width_page_stat_kind::signed_int;
  stats.min_signed = static_cast<int64_t>(
    static_cast<cudf::numeric_scalar<T> const&>(min_scalar).value(stream));
  stats.max_signed = static_cast<int64_t>(
    static_cast<cudf::numeric_scalar<T> const&>(max_scalar).value(stream));
  return stats;
}

template <typename T>
fixed_width_page_stats make_unsigned_page_stats(cudf::scalar const& min_scalar,
                                                cudf::scalar const& max_scalar,
                                                bool has_null,
                                                rmm::cuda_stream_view stream)
{
  fixed_width_page_stats stats;
  stats.valid        = true;
  stats.has_null     = has_null;
  stats.kind         = fixed_width_page_stat_kind::unsigned_int;
  stats.min_unsigned = static_cast<uint64_t>(
    static_cast<cudf::numeric_scalar<T> const&>(min_scalar).value(stream));
  stats.max_unsigned = static_cast<uint64_t>(
    static_cast<cudf::numeric_scalar<T> const&>(max_scalar).value(stream));
  return stats;
}

template <typename T>
fixed_width_page_stats make_floating_page_stats(cudf::scalar const& min_scalar,
                                                cudf::scalar const& max_scalar,
                                                bool has_null,
                                                rmm::cuda_stream_view stream)
{
  fixed_width_page_stats stats;
  stats.valid       = true;
  stats.has_null    = has_null;
  stats.kind        = fixed_width_page_stat_kind::floating;
  stats.min_floating = static_cast<double>(
    static_cast<cudf::numeric_scalar<T> const&>(min_scalar).value(stream));
  stats.max_floating = static_cast<double>(
    static_cast<cudf::numeric_scalar<T> const&>(max_scalar).value(stream));
  return stats;
}

template <typename Timestamp>
fixed_width_page_stats make_timestamp_page_stats(cudf::scalar const& min_scalar,
                                                 cudf::scalar const& max_scalar,
                                                 bool has_null)
{
  fixed_width_page_stats stats;
  stats.valid      = true;
  stats.has_null   = has_null;
  stats.kind       = fixed_width_page_stat_kind::signed_int;
  stats.min_signed = static_cast<int64_t>(
    static_cast<cudf::timestamp_scalar<Timestamp> const&>(min_scalar)
      .value()
      .time_since_epoch()
      .count());
  stats.max_signed = static_cast<int64_t>(
    static_cast<cudf::timestamp_scalar<Timestamp> const&>(max_scalar)
      .value()
      .time_since_epoch()
      .count());
  return stats;
}

fixed_width_page_stats compute_fixed_width_page_stats(cudf::column_view const& page_view,
                                                      rmm::cuda_stream_view stream)
{
  fixed_width_page_stats stats;
  stats.has_null = page_view.has_nulls();
  if (page_view.is_empty()) { return stats; }

  switch (page_view.type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: break;
    default: return stats;
  }

  auto [min_scalar, max_scalar] = cudf::minmax(page_view, stream);
  if (!min_scalar || !max_scalar || !min_scalar->is_valid(stream) ||
      !max_scalar->is_valid(stream)) {
    return stats;
  }

  switch (page_view.type().id()) {
    case cudf::type_id::INT8:
      return make_signed_page_stats<int8_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::INT16:
      return make_signed_page_stats<int16_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::INT32:
      return make_signed_page_stats<int32_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::INT64:
      return make_signed_page_stats<int64_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::UINT8:
      return make_unsigned_page_stats<uint8_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::UINT16:
      return make_unsigned_page_stats<uint16_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::UINT32:
      return make_unsigned_page_stats<uint32_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::UINT64:
      return make_unsigned_page_stats<uint64_t>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::FLOAT32:
      return make_floating_page_stats<float>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::FLOAT64:
      return make_floating_page_stats<double>(*min_scalar, *max_scalar, stats.has_null, stream);
    case cudf::type_id::TIMESTAMP_DAYS:
      return make_timestamp_page_stats<cudf::timestamp_D>(*min_scalar, *max_scalar, stats.has_null);
    case cudf::type_id::TIMESTAMP_SECONDS:
      return make_timestamp_page_stats<cudf::timestamp_s>(*min_scalar, *max_scalar, stats.has_null);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return make_timestamp_page_stats<cudf::timestamp_ms>(*min_scalar, *max_scalar, stats.has_null);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return make_timestamp_page_stats<cudf::timestamp_us>(*min_scalar, *max_scalar, stats.has_null);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return make_timestamp_page_stats<cudf::timestamp_ns>(*min_scalar, *max_scalar, stats.has_null);
    default: return stats;
  }
}

fixed_width_page_stats compute_fixed_width_page_stats_for_range(
  cudf::column_view const& view,
  cudf::size_type begin,
  cudf::size_type end,
  cucascade::memory::memory_space* memory_space)
{
  auto compute = [&]() {
    auto page_views = cudf::slice(view, {begin, end});
    if (page_views.empty()) { return fixed_width_page_stats{}; }
    return compute_fixed_width_page_stats(page_views.front(), cudf::get_default_stream());
  };

  if (memory_space != nullptr && memory_space->get_device_id() >= 0) {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{memory_space->get_device_id()}};
    return compute();
  }
  return compute();
}

std::string fixed_width_page_directory_key_string(fixed_width_page_directory_key const& key)
{
  return key.table_name + "|" + key.file_path + "|" + key.column_name + "|" +
         std::to_string(key.chunk_index) + "|" + std::to_string(key.page_index) + "|" +
         std::to_string(key.device_id);
}

void index_fixed_width_column_pages(pinned_entry& entry,
                                    std::string const& table_name,
                                    std::string const& file_path,
                                    std::string const& column_name,
                                    cudf::column_view view,
                                    std::size_t chunk_index,
                                    std::size_t chunk_global_row_offset,
                                    cucascade::memory::memory_space* memory_space,
                                    std::size_t page_size_bytes,
                                    bool own_page_storage)
{
  if (!is_fixed_width_page_candidate(view) || view.size() <= 0) { return; }

  auto const element_size = cudf::size_of(view.type());
  if (element_size == 0) { return; }

  auto const rows = static_cast<std::size_t>(view.size());
  auto const view_offset_rows =
    static_cast<std::size_t>(std::max<cudf::size_type>(view.offset(), 0));
  auto const rows_per_page = std::max<std::size_t>(1, page_size_bytes / element_size);
  auto& pages              = entry.fixed_width_pages_by_column[column_name];

  for (std::size_t row_offset = 0; row_offset < rows; row_offset += rows_per_page) {
    auto const page_rows = std::min(rows_per_page, rows - row_offset);
    fixed_width_column_page page;
    page.key.table_name    = table_name;
    page.key.file_path     = file_path;
    page.key.column_name   = column_name;
    page.key.chunk_index   = chunk_index;
    page.key.page_index    = pages.size();
    page.key.device_id     = memory_space != nullptr ? memory_space->get_device_id() : -1;
    page.state             = fixed_width_page_state::resident;
    page.chunk_index        = chunk_index;
    page.page_index         = pages.size();
    page.global_row_offset  = chunk_global_row_offset + row_offset;
    page.row_offset         = row_offset;
    page.num_rows           = page_rows;
    page.byte_offset        = (view_offset_rows + row_offset) * element_size;
    page.num_bytes          = page_rows * element_size;
    page.element_size_bytes = element_size;
    page.type_id            = view.type().id();
    page.memory_space       = memory_space;
    auto const begin        = static_cast<cudf::size_type>(row_offset);
    auto const end          = static_cast<cudf::size_type>(row_offset + page_rows);
    page.stats             = compute_fixed_width_page_stats_for_range(
      view, begin, end, memory_space);
    page.admission_score   = fixed_width_page_admission_score(column_name, page.type_id, page.stats);
    page.last_access_tick.store(next_fixed_width_page_lru_tick(), std::memory_order_relaxed);
    if (own_page_storage) {
      auto make_owned_page = [&]() {
        auto sliced = cudf::slice(view, {begin, end}, cudf::get_default_stream());
        if (!sliced.empty()) {
          page.owned_column = std::make_shared<cudf::column>(
            sliced.front(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
        }
      };
      if (memory_space != nullptr && memory_space->get_device_id() >= 0) {
        rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{memory_space->get_device_id()}};
        make_owned_page();
      } else {
        make_owned_page();
      }
    }
    auto const directory_key = fixed_width_page_directory_key_string(page.key);
    entry.fixed_width_page_directory[directory_key] =
      fixed_width_page_directory_entry{column_name, pages.size()};
    pages.emplace_back(page);
  }
}

std::size_t fixed_width_page_count(pinned_entry const& entry)
{
  std::size_t pages = 0;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    pages += col_pages.size();
  }
  return pages;
}

std::size_t fixed_width_page_stats_count(pinned_entry const& entry)
{
  std::size_t pages = 0;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : col_pages) {
      if (page.stats.valid) { ++pages; }
    }
  }
  return pages;
}

fixed_width_page_directory_metrics compute_fixed_width_page_directory_metrics(
  pinned_entry const& entry)
{
  fixed_width_page_directory_metrics metrics;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : col_pages) {
      if (page.state == fixed_width_page_state::resident) {
        ++metrics.resident_pages;
        metrics.resident_bytes += page.num_bytes;
      } else {
        ++metrics.evicted_pages;
      }
      if (page.stats.valid) { ++metrics.stats_pages; }
    }
  }
  return metrics;
}


void apply_fixed_width_page_budget(pinned_entry& entry, std::string const& name)
{
  auto const budget = fixed_page_cache_budget_bytes_per_gpu();
  if (budget == 0) { return; }
  auto const target_budget = fixed_page_cache_low_watermark_bytes_per_gpu(budget);

  std::unordered_map<int, std::size_t> resident_bytes_by_device;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : col_pages) {
      if (page.state != fixed_width_page_state::resident || page.key.device_id < 0) { continue; }
      resident_bytes_by_device[page.key.device_id] += page.num_bytes;
    }
  }

  struct eviction_candidate {
    fixed_width_column_page* page{nullptr};
    std::uint64_t last_access_tick{0};
  };

  std::unordered_map<int, std::vector<eviction_candidate>> candidates_by_device;
  for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto& page : col_pages) {
      if (page.state != fixed_width_page_state::resident || page.key.device_id < 0 ||
          fixed_width_page_is_active(page)) {
        continue;
      }
      candidates_by_device[page.key.device_id].push_back(
        eviction_candidate{&page, page.last_access_tick.load(std::memory_order_relaxed)});
    }
  }

  std::size_t evicted_pages = 0;
  std::size_t evicted_bytes = 0;
  std::uint64_t min_evicted_tick = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_evicted_tick = 0;
  for (auto& [device_id, candidates] : candidates_by_device) {
    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      if (lhs.last_access_tick != rhs.last_access_tick) {
        return lhs.last_access_tick < rhs.last_access_tick;
      }
      if (lhs.page->key.chunk_index != rhs.page->key.chunk_index) {
        return lhs.page->key.chunk_index < rhs.page->key.chunk_index;
      }
      return lhs.page->key.page_index < rhs.page->key.page_index;
    });
    auto& device_bytes = resident_bytes_by_device[device_id];
    for (auto const& candidate : candidates) {
      if (device_bytes <= target_budget) { break; }
      auto& page = *candidate.page;
      if (page.state != fixed_width_page_state::resident || fixed_width_page_is_active(page)) {
        continue;
      }
      page.state = fixed_width_page_state::evicted;
      page.owned_column.reset();
      device_bytes -= std::min(device_bytes, page.num_bytes);
      ++entry.fixed_width_page_metrics.eviction_count;
      ++evicted_pages;
      evicted_bytes += page.num_bytes;
      min_evicted_tick = std::min(min_evicted_tick, candidate.last_access_tick);
      max_evicted_tick = std::max(max_evicted_tick, candidate.last_access_tick);
    }
  }

  if (evicted_pages != 0) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] page_budget applied table='{}' policy=lru budget_bytes_per_gpu={} "
      "target_bytes_per_gpu={} evicted_pages={} evicted_bytes={} min_last_access_tick={} "
      "max_last_access_tick={}",
      name,
      budget,
      target_budget,
      evicted_pages,
      evicted_bytes,
      min_evicted_tick,
      max_evicted_tick);
  }
}

void apply_global_fixed_width_page_budget(std::unordered_map<std::string, pinned_entry>& entries)
{
  auto const budget = fixed_page_cache_budget_bytes_per_gpu();
  if (budget == 0) { return; }
  auto const target_budget = fixed_page_cache_low_watermark_bytes_per_gpu(budget);

  std::unordered_map<int, std::size_t> resident_bytes_by_device;
  struct eviction_candidate {
    pinned_entry* entry{nullptr};
    fixed_width_column_page* page{nullptr};
    std::string table_name;
    std::uint64_t last_access_tick{0};
  };
  std::unordered_map<int, std::vector<eviction_candidate>> candidates_by_device;

  for (auto& [entry_name, entry] : entries) {
    for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
      for (auto& page : col_pages) {
        if (page.state != fixed_width_page_state::resident || page.key.device_id < 0) { continue; }
        resident_bytes_by_device[page.key.device_id] += page.num_bytes;
        if (fixed_width_page_is_active(page)) { continue; }
        candidates_by_device[page.key.device_id].push_back(
          eviction_candidate{&entry,
                             &page,
                             entry_name,
                             page.last_access_tick.load(std::memory_order_relaxed)});
      }
    }
  }

  for (auto& [device_id, candidates] : candidates_by_device) {
    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      if (lhs.last_access_tick != rhs.last_access_tick) {
        return lhs.last_access_tick < rhs.last_access_tick;
      }
      if (lhs.table_name != rhs.table_name) { return lhs.table_name < rhs.table_name; }
      if (lhs.page->key.column_name != rhs.page->key.column_name) {
        return lhs.page->key.column_name < rhs.page->key.column_name;
      }
      if (lhs.page->key.chunk_index != rhs.page->key.chunk_index) {
        return lhs.page->key.chunk_index < rhs.page->key.chunk_index;
      }
      return lhs.page->key.page_index < rhs.page->key.page_index;
    });

    auto& device_bytes = resident_bytes_by_device[device_id];
    std::size_t evicted_pages = 0;
    std::size_t evicted_bytes = 0;
    std::uint64_t min_evicted_tick = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_evicted_tick = 0;
    for (auto const& candidate : candidates) {
      if (device_bytes <= target_budget) { break; }
      auto& page = *candidate.page;
      if (page.state != fixed_width_page_state::resident || fixed_width_page_is_active(page)) {
        continue;
      }
      page.state = fixed_width_page_state::evicted;
      page.owned_column.reset();
      device_bytes -= std::min(device_bytes, page.num_bytes);
      ++candidate.entry->fixed_width_page_metrics.eviction_count;
      ++evicted_pages;
      evicted_bytes += page.num_bytes;
      min_evicted_tick = std::min(min_evicted_tick, candidate.last_access_tick);
      max_evicted_tick = std::max(max_evicted_tick, candidate.last_access_tick);
    }

    if (evicted_pages != 0) {
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_budget applied scope=global device={} policy=lru "
        "budget_bytes_per_gpu={} target_bytes_per_gpu={} evicted_pages={} evicted_bytes={} "
        "resident_bytes_after={} min_last_access_tick={} max_last_access_tick={}",
        device_id,
        budget,
        target_budget,
        evicted_pages,
        evicted_bytes,
        device_bytes,
        min_evicted_tick,
        max_evicted_tick);
    }
  }

  for (auto& [_, entry] : entries) {
    auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
    entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
    entry.fixed_width_page_metrics.eviction_count = eviction_count;
  }
}


std::size_t fixed_width_page_resident_alloc_bytes(fixed_width_column_page const& page)
{
  if (page.owned_column) {
    return std::max<std::size_t>(page.num_bytes, page.owned_column->alloc_size());
  }
  return page.num_bytes;
}

void apply_global_fixed_width_page_memory_pressure(
  std::unordered_map<std::string, pinned_entry>& entries,
  std::string_view reason)
{
  auto const min_free = fixed_page_cache_min_free_bytes_per_gpu();
  if (min_free == 0) { return; }

  struct eviction_candidate {
    pinned_entry* entry{nullptr};
    fixed_width_column_page* page{nullptr};
    std::string table_name;
    std::uint64_t last_access_tick{0};
  };
  std::unordered_map<int, std::vector<eviction_candidate>> candidates_by_device;

  for (auto& [entry_name, entry] : entries) {
    for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
      for (auto& page : col_pages) {
        if (page.state != fixed_width_page_state::resident || page.key.device_id < 0 ||
            !page.owned_column || fixed_width_page_is_active(page)) {
          continue;
        }
        candidates_by_device[page.key.device_id].push_back(
          eviction_candidate{&entry,
                             &page,
                             entry_name,
                             page.last_access_tick.load(std::memory_order_relaxed)});
      }
    }
  }

  for (auto& [device_id, candidates] : candidates_by_device) {
    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      if (lhs.last_access_tick != rhs.last_access_tick) {
        return lhs.last_access_tick < rhs.last_access_tick;
      }
      if (lhs.table_name != rhs.table_name) { return lhs.table_name < rhs.table_name; }
      if (lhs.page->key.column_name != rhs.page->key.column_name) {
        return lhs.page->key.column_name < rhs.page->key.column_name;
      }
      if (lhs.page->key.chunk_index != rhs.page->key.chunk_index) {
        return lhs.page->key.chunk_index < rhs.page->key.chunk_index;
      }
      return lhs.page->key.page_index < rhs.page->key.page_index;
    });

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
      auto const status = cudaMemGetInfo(&free_before, &total_bytes);
      if (status != cudaSuccess) {
        SIRIUS_LOG_WARN(
          "[fixed-page-cache] memory_pressure cudaMemGetInfo failed device={} error='{}'",
          device_id,
          cudaGetErrorString(status));
        continue;
      }
    }

    if (free_before >= min_free) { continue; }

    auto const required_bytes = min_free - free_before;
    std::size_t released_alloc_bytes = 0;
    std::size_t evicted_pages        = 0;
    std::size_t evicted_bytes        = 0;
    std::uint64_t min_evicted_tick   = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_evicted_tick   = 0;
    for (auto const& candidate : candidates) {
      if (released_alloc_bytes >= required_bytes) { break; }
      auto& page = *candidate.page;
      if (page.state != fixed_width_page_state::resident || !page.owned_column ||
          fixed_width_page_is_active(page)) {
        continue;
      }
      auto const alloc_bytes = fixed_width_page_resident_alloc_bytes(page);
      page.state = fixed_width_page_state::evicted;
      page.owned_column.reset();
      released_alloc_bytes += alloc_bytes;
      evicted_bytes += page.num_bytes;
      ++candidate.entry->fixed_width_page_metrics.eviction_count;
      ++evicted_pages;
      min_evicted_tick = std::min(min_evicted_tick, candidate.last_access_tick);
      max_evicted_tick = std::max(max_evicted_tick, candidate.last_access_tick);
    }

    if (evicted_pages == 0) { continue; }

    std::size_t free_after = 0;
    {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
      auto const sync_status = cudaDeviceSynchronize();
      if (sync_status != cudaSuccess) {
        SIRIUS_LOG_WARN(
          "[fixed-page-cache] memory_pressure cudaDeviceSynchronize failed device={} error='{}'",
          device_id,
          cudaGetErrorString(sync_status));
      }
      auto const info_status = cudaMemGetInfo(&free_after, &total_bytes);
      if (info_status != cudaSuccess) {
        SIRIUS_LOG_WARN(
          "[fixed-page-cache] memory_pressure post-evict cudaMemGetInfo failed device={} error='{}'",
          device_id,
          cudaGetErrorString(info_status));
        free_after = 0;
      }
    }

    SIRIUS_LOG_INFO(
      "[fixed-page-cache] memory_pressure applied scope=global reason='{}' device={} policy=lru "
      "min_free_bytes_per_gpu={} free_bytes_before={} free_bytes_after={} total_bytes={} "
      "evicted_pages={} evicted_bytes={} evicted_alloc_bytes={} "
      "min_last_access_tick={} max_last_access_tick={}",
      reason,
      device_id,
      min_free,
      free_before,
      free_after,
      total_bytes,
      evicted_pages,
      evicted_bytes,
      released_alloc_bytes,
      min_evicted_tick,
      max_evicted_tick);
  }

  for (auto& [_, entry] : entries) {
    auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
    entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
    entry.fixed_width_page_metrics.eviction_count = eviction_count;
  }
}

void apply_global_fixed_width_page_eviction_policies(
  std::unordered_map<std::string, pinned_entry>& entries,
  std::string_view reason)
{
  apply_global_fixed_width_page_budget(entries);
  apply_global_fixed_width_page_memory_pressure(entries, reason);

  std::size_t true_total = 0;
  for (auto const& [name, entry] : entries) {
    for (auto const& [col_name, pages] : entry.fixed_width_pages_by_column) {
      for (auto const& page : pages) {
        if (page.state == fixed_width_page_state::resident) { true_total += page.num_bytes; }
      }
    }
  }
  SIRIUS_LOG_INFO("[bug-hunt] true_global_resident_bytes={} ({:.3f} GB) reason={} num_entries={}",
                  true_total, true_total / 1e9, reason, entries.size());
}

// wdy end
}  // namespace

sirius_scan_manager::sirius_scan_manager(
  const scan_manager_config& config,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  std::shared_ptr<const sirius::memory::topology_index> topology_index)
  : _config(config),
    _reservation_manager(reservation_manager),
    _topology_index(std::move(topology_index)),
    _thread_pool(_config.thread_pool.num_threads + 1,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _thread_pool.num_threads())),
    _ioctx_registry(config, reservation_manager)
{
  if (!_topology_index) {
    throw std::invalid_argument("[sirius_scan_manager] topology_index must be non-null");
  }

  // scan_manager always owns an io_ctx: sirius_datasource (uring) on the
  // fast path, kvikio_context as the universal fallback so the rest of the
  // scan path (parquet_split_provider, scan tasks) always has an ioctx to
  // talk to.  kvikio_context wraps cudf::io::datasource so the read path
  // is identical from the caller's point of view.  Both are built by the
  // ioctx registry, which sources the reactor staging resource from the
  // reservation manager it was constructed with.
  if (_config.use_sirius_datasource) {
    _io_ctx = _ioctx_registry.make_ioctx(sirius::io::io_context_type::uring);
    if (!_io_ctx) {
      throw std::runtime_error("[sirius_scan_manager] failed to create uring io_context");
    }
    SIRIUS_LOG_DEBUG("[sirius_scan_manager] sirius_datasource enabled (uring_ioctx n_reactors={})",
                     _config.uring_n_reactors);
  } else {
    if (_topology_index->gpu_ids().size() > 1) {
      throw std::runtime_error(
        "[sirius_scan_manager] kvikio_context fallback (use_sirius_datasource=false) "
        "does not support multi-GPU; topology reports " +
        std::to_string(_topology_index->gpu_ids().size()) +
        " GPUs.  Enable use_sirius_datasource for multi-GPU runs.");
    }
    _io_ctx = _ioctx_registry.make_ioctx(sirius::io::io_context_type::kvikio);
    if (!_io_ctx) {
      throw std::runtime_error("[sirius_scan_manager] failed to create kvikio io_context");
    }
    SIRIUS_LOG_DEBUG(
      "[sirius_scan_manager] sirius_datasource disabled — using kvikio_context fallback");
  }

  // Build the prefetching cache on the ioctx.  Budget=0 keeps the
  // cache unarmed (no background threads); we pass that whenever the
  // user has disabled prefetching so the construction is always
  // unconditional and there's no "is the cache present" branch to
  // worry about in callers.
  if (_config.enable_prefetch_cache && _io_ctx->can_use_prefetching_cache()) {
    _io_ctx->initialize_cache(reservation_manager, _config.cache, _topology_index);
  }

  // Reactors are built parked; start() launches their worker threads and
  // allocates per-reactor staging.  No-op for the kvikio fallback (no reactors).
  _io_ctx->start();
}

sirius_scan_manager::~sirius_scan_manager()
{
  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
  }
  // Drain the dispatcher (and the worker pool) first so no in-flight
  // metadata-scan / sequencer task can still be reaching into the
  // cache via _io_ctx when we tear it down below.
  stop();
  // Tear down the cache (which owns its buffer_pool).  shutdown_cache drains
  // in-flight IO before the pool is destroyed, so callbacks release their
  // chunks safely.
  if (_io_ctx) { _io_ctx->shutdown_cache(); }
  // Same drain for any path-routed ioctxs; their reactors stop when the
  // shared_ptrs in _routed_io_ctxs are released (member destruction below).
  std::lock_guard lk{_routed_io_ctxs_mtx};
  for (auto& [type, io_ctx] : _routed_io_ctxs) {
    if (io_ctx) { io_ctx->shutdown_cache(); }
  }
}

parquet_bind_result sirius_scan_manager::describe_parquet(std::string const& uri)
{
  auto datasource = create_datasource(uri);
  if (!datasource) {
    throw std::runtime_error("[sirius_scan_manager::describe_parquet] no backend supports URI: " +
                             uri);
  }

  // Reuse a previously parsed footer when present — a prior bind or scan of the
  // same file parks it in the ioctx metadata store, which lives for the ioctx's
  // lifetime. On a miss, fetch + Thrift-parse the footer once and park it so the
  // subsequent scan reuses it. Mirrors parquet_gpu_ingestible::build_file_scan_info,
  // so the footer is parsed exactly once per file per process.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  if (auto cached = datasource->metadata()) {
    if (auto pm = std::dynamic_pointer_cast<op::scan::parquet_metadata>(std::move(cached))) {
      file_metadata = pm->file_metadata();
    }
  }
  if (!file_metadata) {
    auto footer_buffer         = cudf::io::parquet::fetch_footer_to_host(*datasource);
    auto const footer_byte_len = footer_buffer->size();
    auto reader_options        = cudf::io::parquet_reader_options::builder().build();
    cudf::io::parquet::experimental::hybrid_scan_reader reader{
      cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()),
      reader_options};
    file_metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(reader.parquet_metadata());
    [[maybe_unused]] auto const stored = datasource->store_metadata(
      std::make_shared<op::scan::parquet_metadata>(file_metadata, footer_byte_len));
  }

  auto schema = sirius::io::parquet_helpers::extract_schema(*file_metadata);

  parquet_bind_result result;
  result.return_types   = std::move(schema.types);
  result.names          = std::move(schema.names);
  result.object_size    = datasource->size();
  result.total_num_rows = static_cast<std::size_t>(file_metadata->num_rows);
  return result;
}


void sirius_scan_manager::evict_fixed_pages_for_memory_pressure(std::string_view reason)
{
  std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
  apply_global_fixed_width_page_memory_pressure(_pinned_entries, reason);
}

void sirius_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  if (_io_ctx && _io_ctx->cache()) {
    SIRIUS_LOG_INFO("[sirius_scan_manager] cache summary: {}", _io_ctx->cache()->summary());
    _io_ctx->cache()->prepare_for_query(query);
  }

  // Routed ioctxs (e.g. the restful context serving s3://) are built lazily and
  // reused across queries; advance their caches to this query too, or a routed
  // cache's epoch freezes at build time and a later query serves the prior
  // query's cached chunks as current.
  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    for (auto& [type, io_ctx] : _routed_io_ctxs) {
      if (io_ctx && io_ctx->cache()) { io_ctx->cache()->prepare_for_query(query); }
    }
  }

  evict_fixed_pages_for_memory_pressure("prepare_for_query");

  auto const gpu_ids = _topology_index->gpu_ids();
  auto round_robin =
    std::make_shared<round_robin_strategy>(std::vector<int>(gpu_ids.begin(), gpu_ids.end()));

  _metadata_processor = std::make_unique<load_balancing_scan_batch_coalescer>();

  for (auto const& scan_op : query.get_scan_operators()) {
    if (scan_op->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }
    auto* op = &scan_op->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }
    _metadata_processor->register_pipeline(op, round_robin);
    if (auto* parquet = dynamic_cast<op::scan::parquet_gpu_ingestible*>(&op->get_ingestible())) {
      parquet->set_scan_manager(this);
    }
    // On a pinned-cache hit the coalescer serves this operator from the cached
    // batch_provider (process_cached_entries); skip the disk-reading
    // split_provider entirely so no read is issued for the cached scan.
    if (try_assign_cached_entries(op)) {
      _scan_op_order.push_back(op);
      continue;
    }
    auto provider = std::make_unique<split_provider>(
      op->get_ingestible(),
      [this](std::string_view file_path) -> std::shared_ptr<io::sirius_ioctx> {
        auto io_ctx = ioctx_for_path(file_path);
        if (!io_ctx) {
          throw std::runtime_error("scan_manager: no backend supports path: " +
                                   std::string(file_path));
        }
        return io_ctx;
      });
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);
  }

  if (_scan_op_order.empty()) {
    spdlog::warn("[sirius_scan_manager::prepare_for_query] no GPU scan operators found in query");
    return;
  }

  start_metadata_processing();
}

void sirius_scan_manager::start_metadata_processing()
{
  _metadata_processor->spawn_workers(*_dispatcher);
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    it->second->run(*_dispatcher, _metadata_processor->get_split_provider_bridge(op));
  }
}

std::shared_ptr<sirius::io::sirius_datasource> sirius_scan_manager::create_datasource(
  std::string_view path)
{
  auto file_path = normalize_path(std::string(path));
  auto io_ctx    = ioctx_for_path(file_path);
  if (!io_ctx) { return nullptr; }  // no backend supports the path
  // Real I/O / HEAD / auth / missing-object errors propagate as exceptions;
  // only "no backend" is reported as nullptr (callers map it to that message).
  return io_ctx->open_datasource(file_path);
}

std::shared_ptr<sirius::io::sirius_ioctx> sirius_scan_manager::ioctx_for_path(std::string_view path)
{
  // Normalize here so every caller (incl. the scan resolver, which forwards raw
  // ingestible paths) routes `file://` the same way create_datasource does.
  auto file_path = normalize_path(std::string(path));
  auto type      = _ioctx_registry.lookup_path(file_path);
  if (!type) { return nullptr; }
  // The local default `_io_ctx` already serves uring/kvikio; only an off-default
  // backend (e.g. s3:// -> restful) needs a separate, lazily-built context.
  if (_io_ctx && _io_ctx->type() == *type) { return _io_ctx; }

  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    if (auto it = _routed_io_ctxs.find(*type); it != _routed_io_ctxs.end()) { return it->second; }
  }
  // Build outside the map mutex: make_ioctx/start spawn reactor threads and
  // initialize_cache allocates, so holding _routed_io_ctxs_mtx across them would
  // park every concurrent lookup behind one long critical section. The build
  // mutex serializes builders instead, so two first-touches of the same type
  // never construct twice (a losing ioctx would need drain/stop teardown).
  std::lock_guard build_lk{_routed_io_ctxs_build_mtx};
  {
    std::lock_guard lk{_routed_io_ctxs_mtx};
    if (auto it = _routed_io_ctxs.find(*type); it != _routed_io_ctxs.end()) { return it->second; }
  }
  auto io_ctx = _ioctx_registry.make_ioctx(*type);
  if (!io_ctx) { return nullptr; }
  io_ctx->start();
  if (_config.enable_prefetch_cache && io_ctx->can_use_prefetching_cache()) {
    io_ctx->initialize_cache(_reservation_manager, _config.cache, _topology_index);
  }
  std::lock_guard lk{_routed_io_ctxs_mtx};
  auto [it, inserted] = _routed_io_ctxs.emplace(*type, std::move(io_ctx));
  return it->second;
}

void sirius_scan_manager::reset()
{
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  _metadata_processor.reset();
  _dispatcher = std::make_unique<exec::scoped_dispatcher>(_thread_pool, _thread_pool.num_threads());
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  _thread_pool.stop();
}

namespace {

// Gather positions into @p cached_ids for each requested primary (storage) index, in the
// given order. Empty when any requested column is absent — i.e. the cache is not a superset.
std::vector<std::size_t> gather_by_primary_index(
  duckdb::vector<duckdb::ColumnIndex> const& cached_ids,
  std::vector<std::size_t> const& requested_primary_indices)
{
  std::unordered_map<duckdb::idx_t, std::size_t> pos;
  pos.reserve(cached_ids.size());
  for (std::size_t i = 0; i < cached_ids.size(); ++i) {
    pos.emplace(cached_ids[i].GetPrimaryIndex(), i);
  }
  std::vector<std::size_t> projection;
  projection.reserve(requested_primary_indices.size());
  for (auto const primary_idx : requested_primary_indices) {
    auto it = pos.find(primary_idx);
    if (it == pos.end()) { return {}; }  // cache lacks a requested column
    projection.push_back(it->second);
  }
  return projection;
}

// Gather projection that lets a cache holding @p cached_ids (by primary/storage
// index) serve a scan requesting @p requested_ids: for each requested column,
// its position within @p cached_ids, in the requested order. Empty when any
// requested column is absent — i.e. the cache is not a column superset.
std::vector<std::size_t> column_superset_projection(
  duckdb::vector<duckdb::ColumnIndex> const& cached_ids,
  duckdb::vector<duckdb::ColumnIndex> const& requested_ids)
{
  std::vector<std::size_t> requested_primary_indices;
  requested_primary_indices.reserve(requested_ids.size());
  for (auto const& c : requested_ids) {
    requested_primary_indices.push_back(c.GetPrimaryIndex());
  }
  return gather_by_primary_index(cached_ids, requested_primary_indices);
}

// column_ids-aligned names: for each column_ids[i], the full-schema name at its
// primary (storage) index — the keys data_batches_by_column / the gather use.
std::vector<std::string> aligned_column_names(duckdb::vector<std::string> const& full_names,
                                              duckdb::vector<duckdb::ColumnIndex> const& column_ids)
{
  std::vector<std::string> out;
  out.reserve(column_ids.size());
  for (auto const& c : column_ids) {
    auto const p = static_cast<std::size_t>(c.GetPrimaryIndex());
    out.push_back(p < full_names.size() ? full_names[p] : std::string{});
  }
  return out;
}

}  // namespace

cache_entry_info cache_entry_info::from(const op::scan::ingestible_table_info& info)
{
  cache_entry_info ci;
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&info)) {
    ci.resolved_file_paths = p->resolved_file_paths;
    ci.column_ids          = p->column_ids;
    ci.names               = aligned_column_names(p->names, p->column_ids);
  } else if (auto const* d =
               dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&info)) {
    ci.catalog_name = d->catalog_name;
    ci.schema_name  = d->schema_name;
    ci.table_name   = d->table_name;
    ci.column_ids   = d->column_ids;
    ci.names        = aligned_column_names(d->names, d->column_ids);
  }
  return ci;
}

std::vector<std::size_t> cache_entry_info::can_serve_with_columns(
  const op::scan::ingestible_table_info& other) const
{
  // A parquet pin serves a parquet scan over the same file set; a duckdb pin
  // serves a duckdb scan over the same catalog.schema.table. A cache of one format
  // never serves a scan of the other — the identity check below falls through (a
  // duckdb cache has empty resolved_file_paths; a parquet cache has an empty table_name).
  if (auto const* p = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&other)) {
    if (resolved_file_paths.size() != p->resolved_file_paths.size()) { return {}; }
    auto these_files = resolved_file_paths;
    auto those_files = p->resolved_file_paths;
    std::sort(these_files.begin(), these_files.end());
    std::sort(those_files.begin(), those_files.end());
    if (these_files != those_files) { return {}; }
    return column_superset_projection(column_ids, p->column_ids);
  }
  if (auto const* d = dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&other)) {
    // Same duckdb table by qualified name (catalog.schema.table), derived on both
    // pin and query sides from the resolved DuckTableEntry — so the stored casing is
    // the table's canonical (case-preserved) name on both sides and a byte-exact
    // compare is correct. (If a future site ever populates these from parsed input
    // rather than the resolved entry, switch to a case-insensitive compare.)
    // A parquet cache has an empty table_name, so it never matches a duckdb scan.
    if (table_name.empty()) { return {}; }
    if (catalog_name != d->catalog_name || schema_name != d->schema_name ||
        table_name != d->table_name) {
      return {};
    }
    return column_superset_projection(column_ids, d->column_ids);
  }
  return {};
}

bool sirius_scan_manager::insert_fixed_page_entry_from_view(
  const std::string& name,
  cache_entry_info cache_info,
  cudf::table_view view,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream)
{
  (void)stream;
  if (view.num_columns() <= 0 || view.num_rows() <= 0) { return false; }

  std::vector<std::string> column_names = cache_info.column_names();
  if (cache_info.column_ids.size() != column_names.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_fixed_page_entry_from_view] cache_info column_ids/names "
      "size mismatch");
  }
  if (column_names.size() != static_cast<std::size_t>(view.num_columns())) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_fixed_page_entry_from_view] table column count " +
      std::to_string(view.num_columns()) + " does not match column_names size " +
      std::to_string(column_names.size()));
  }
  bool const hybrid_provider = fixed_page_hybrid_provider_enabled();
  std::vector<bool> fixed_columns;
  fixed_columns.reserve(static_cast<std::size_t>(view.num_columns()));
  bool has_fixed_width_column = false;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    bool const fixed_width = is_fixed_width_page_candidate(view.column(i));
    fixed_columns.push_back(fixed_width);
    has_fixed_width_column = has_fixed_width_column || fixed_width;
    if (!fixed_width && !hybrid_provider) {
      throw std::invalid_argument(
        "[sirius_scan_manager::insert_fixed_page_entry_from_view] non fixed-width column");
    }
  }
  if (!has_fixed_width_column) { return false; }

  std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
  apply_global_fixed_width_page_memory_pressure(_pinned_entries, "pre_fixed_page_insert");

  auto const admission_limit = fixed_page_admission_max_entry_bytes();
  auto const incoming_bytes  = fixed_width_table_view_bytes(view);
  auto reject_admission = [&](std::size_t existing_bytes, std::size_t projected_bytes) {
    _pinned_entries.erase(name);
    _fixed_page_admission_rejected_entries.insert(name);
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] auto_cache_skip reason=admission_entry_bytes table='{}' "
      "existing_bytes={} incoming_bytes={} projected_bytes={} max_entry_bytes={}",
      name,
      existing_bytes,
      incoming_bytes,
      projected_bytes,
      admission_limit);
    return false;
  };

  if (_fixed_page_admission_rejected_entries.contains(name)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] auto_cache_skip reason=admission_previously_rejected table='{}' "
      "incoming_bytes={} max_entry_bytes={}",
      name,
      incoming_bytes,
      admission_limit);
    return false;
  }

  if (admission_limit != 0 && incoming_bytes > admission_limit) {
    return reject_admission(0, incoming_bytes);
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    auto& entry = existing_it->second;
    bool const appendable = is_auto_fixed_page_entry(name) &&
                            entry.cache_info.column_ids.size() == cache_info.column_ids.size() &&
                            entry.cache_info.names == column_names &&
                            entry.cache_info.filter_signature == cache_info.filter_signature;
    if (!appendable) {
      _pinned_entries.erase(existing_it);
      existing_it = _pinned_entries.end();
    } else if (static_cast<std::size_t>(view.num_rows()) == entry.num_rows) {
      // Two concurrent scan tasks can both miss the cache for the same brand-new
      // (name, filter) key, both read it from disk, and both land here to populate
      // it. The first call legitimately creates the entry; without this guard the
      // second call's "appendable" branch below treats its own from-scratch copy of
      // the SAME rows as a genuinely new chunk and appends it, silently doubling
      // entry.num_rows (and every future query's join/aggregate output) for that
      // table. A real incremental chunk of a large table essentially never has a
      // row count that exactly equals the running total accumulated so far, so
      // this is a safe, cheap signature for "this is a duplicate of what we
      // already have," not a genuinely new chunk to fold in.
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] auto_cache_skip reason=duplicate_concurrent_populate table='{}' "
        "incoming_rows={} existing_rows={}",
        name,
        view.num_rows(),
        entry.num_rows);
      return true;
    } else {
      auto const existing_bytes  = fixed_width_entry_logical_bytes(entry);
      auto const projected_bytes = existing_bytes + incoming_bytes;
      if (admission_limit != 0 && projected_bytes > admission_limit) {
        return reject_admission(existing_bytes, projected_bytes);
      }
      if (entry.fixed_width_page_size_bytes == 0) {
        entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();
      }
      auto const chunk_index = entry.chunk_memory_spaces.size();
      entry.chunk_memory_spaces.push_back(&memory_space);
      for (std::size_t i = 0; i < column_names.size(); ++i) {
        auto& chunks = entry.data_batches_by_column[std::string{column_names[i]}];
        auto const column_view = view.column(static_cast<cudf::size_type>(i));
        if (!fixed_columns[i]) {
          chunks.emplace_back(std::make_shared<cudf::column>(
            column_view, stream, memory_space.get_default_allocator()));
          continue;
        }

        std::size_t chunk_global_row_offset = 0;
        auto pages_it = entry.fixed_width_pages_by_column.find(column_names[i]);
        if (pages_it != entry.fixed_width_pages_by_column.end()) {
          for (auto const& page : pages_it->second) {
            chunk_global_row_offset =
              std::max(chunk_global_row_offset, page.global_row_offset + page.num_rows);
          }
        }
        index_fixed_width_column_pages(entry,
                                       name,
                                       entry.cache_info.resolved_file_paths.empty()
                                         ? std::string{}
                                         : entry.cache_info.resolved_file_paths.front(),
                                       column_names[i],
                                       column_view,
                                       chunk_index,
                                       chunk_global_row_offset,
                                       &memory_space,
                                       entry.fixed_width_page_size_bytes,
                                       true);
        chunks.emplace_back(nullptr);
      }
      entry.num_rows += static_cast<std::size_t>(view.num_rows());
      entry.fixed_width_page_metrics = {};
      apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
      auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
      entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
      entry.fixed_width_page_metrics.eviction_count = eviction_count;
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_directory direct_appended table='{}' fixed_cols={} pages={} "
        "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} eviction_count={} "
        "directory_entries={} page_bytes={} rows={} owned_pages=1",
        name,
        entry.fixed_width_pages_by_column.size(),
        fixed_width_page_count(entry),
        entry.fixed_width_page_metrics.resident_pages,
        entry.fixed_width_page_metrics.resident_bytes,
        entry.fixed_width_page_metrics.stats_pages,
        entry.fixed_width_page_metrics.evicted_pages,
        entry.fixed_width_page_metrics.eviction_count,
        entry.fixed_width_page_directory.size(),
        entry.fixed_width_page_size_bytes,
        entry.num_rows);
      return true;
    }
  }

  pinned_entry entry;
  entry.cache_info = std::move(cache_info);
  entry.chunk_memory_spaces.push_back(&memory_space);
  entry.tier                        = cucascade::memory::Tier::GPU;
  entry.num_rows                    = static_cast<std::size_t>(view.num_rows());
  entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();

  for (std::size_t i = 0; i < column_names.size(); ++i) {
    auto& chunks = entry.data_batches_by_column[std::string{column_names[i]}];
    auto const column_view = view.column(static_cast<cudf::size_type>(i));
    if (!fixed_columns[i]) {
      chunks.emplace_back(std::make_shared<cudf::column>(
        column_view, stream, memory_space.get_default_allocator()));
      continue;
    }

    index_fixed_width_column_pages(entry,
                                   name,
                                   entry.cache_info.resolved_file_paths.empty()
                                     ? std::string{}
                                     : entry.cache_info.resolved_file_paths.front(),
                                   column_names[i],
                                   column_view,
                                   0,
                                   0,
                                   &memory_space,
                                   entry.fixed_width_page_size_bytes,
                                   true);
    chunks.emplace_back(nullptr);
  }

  entry.fixed_width_page_metrics = {};
  _pinned_entries[name] = std::move(entry);
  auto& inserted_entry = _pinned_entries.at(name);
  apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
  auto eviction_count = inserted_entry.fixed_width_page_metrics.eviction_count;
  inserted_entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(inserted_entry);
  inserted_entry.fixed_width_page_metrics.eviction_count = eviction_count;
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] page_directory direct_indexed table='{}' fixed_cols={} pages={} "
    "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} eviction_count={} "
    "directory_entries={} page_bytes={} rows={} owned_pages=1",
    name,
    inserted_entry.fixed_width_pages_by_column.size(),
    fixed_width_page_count(inserted_entry),
    inserted_entry.fixed_width_page_metrics.resident_pages,
    inserted_entry.fixed_width_page_metrics.resident_bytes,
    inserted_entry.fixed_width_page_metrics.stats_pages,
    inserted_entry.fixed_width_page_metrics.evicted_pages,
    inserted_entry.fixed_width_page_metrics.eviction_count,
    inserted_entry.fixed_width_page_directory.size(),
    inserted_entry.fixed_width_page_size_bytes,
    inserted_entry.num_rows);
  return true;
}

void sirius_scan_manager::insert_pinned_entry(
  const std::string& name,
  cache_entry_info cache_info,
  std::vector<std::unique_ptr<cudf::table>> data_tables,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces)
{
  // chunk_memory_spaces is parallel to data_tables — the caller
  // (PinTableFunction) emits one memory_space* per
  // chunked_parquet_reader::read_chunk() result, and there is exactly one
  // cudf::table per chunk in data_tables. Reject any misalignment loudly
  // rather than silently aliasing chunks to the wrong GPU.
  if (chunk_memory_spaces.size() != data_tables.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry] chunk_memory_spaces.size() (" +
      std::to_string(chunk_memory_spaces.size()) + ") must equal data_tables.size() (" +
      std::to_string(data_tables.size()) + ")");
  }

  // Compute the total row count of the incoming tables before releasing them
  // (release() empties the table; num_rows() would then return 0).
  std::size_t new_num_rows = 0;
  for (auto const& table : data_tables) {
    if (table) { new_num_rows += static_cast<std::size_t>(table->num_rows()); }
  }

  // Column names (aligned with the cached column_ids) key data_batches_by_column.
  // Copied out before cache_info is moved into the entry below.
  std::vector<std::string> column_names = cache_info.column_names();

  // column_ids and names within cache_info are built aligned 1:1 by
  // cache_entry_info::from; the merge path below indexes column_ids by the same
  // position as the column names, so reject any misalignment loudly rather than
  // risk an out-of-bounds access.
  if (cache_info.column_ids.size() != column_names.size()) {
    throw std::invalid_argument(
      "[sirius_scan_manager::insert_pinned_entry] cache_info.column_ids.size() (" +
      std::to_string(cache_info.column_ids.size()) + ") must equal column_names size (" +
      std::to_string(column_names.size()) + ")");
  }

  std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
  apply_global_fixed_width_page_memory_pressure(_pinned_entries, "pre_pinned_insert");

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    if (is_auto_fixed_page_entry(name) &&
        existing_it->second.cache_info.column_ids.size() == cache_info.column_ids.size() &&
        existing_it->second.cache_info.names == column_names &&
        existing_it->second.cache_info.filter_signature == cache_info.filter_signature) {
      auto& entry = existing_it->second;
      if (entry.fixed_width_page_size_bytes == 0) {
        entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();
      }
      bool const own_page_storage = fixed_page_owned_pages_enabled();
      for (std::size_t table_idx = 0; table_idx < data_tables.size(); ++table_idx) {
        auto& table = data_tables[table_idx];
        if (!table) { continue; }
        auto const chunk_index = entry.chunk_memory_spaces.size();
        auto* chunk_space = table_idx < chunk_memory_spaces.size() ? chunk_memory_spaces[table_idx] : nullptr;
        entry.chunk_memory_spaces.push_back(chunk_space);
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          auto column = std::move(cols[i]);
          auto& chunks = entry.data_batches_by_column[std::string{column_names[i]}];
          std::size_t chunk_global_row_offset = 0;
          for (auto const& existing_chunk : chunks) {
            if (existing_chunk) { chunk_global_row_offset += static_cast<std::size_t>(existing_chunk->size()); }
          }
          auto pages_it = entry.fixed_width_pages_by_column.find(column_names[i]);
          if (pages_it != entry.fixed_width_pages_by_column.end()) {
            for (auto const& page : pages_it->second) {
              chunk_global_row_offset = std::max(chunk_global_row_offset, page.global_row_offset + page.num_rows);
            }
          }
          auto const page_file_path = entry.cache_info.resolved_file_paths.empty()
                                      ? std::string{}
                                      : entry.cache_info.resolved_file_paths.front();
          bool const fixed_candidate = is_fixed_width_page_candidate(column->view());
          index_fixed_width_column_pages(entry,
                                         name,
                                         page_file_path,
                                         column_names[i],
                                         column->view(),
                                         chunk_index,
                                         chunk_global_row_offset,
                                         chunk_space,
                                         entry.fixed_width_page_size_bytes,
                                         own_page_storage && fixed_candidate);
          if (own_page_storage && fixed_candidate) {
            chunks.emplace_back(nullptr);
          } else {
            chunks.emplace_back(std::move(column));
          }
        }
      }
      entry.num_rows += new_num_rows;
      entry.fixed_width_page_metrics = {};
      apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
      auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
      entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
      entry.fixed_width_page_metrics.eviction_count = eviction_count;
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_directory appended table='{}' fixed_cols={} pages={} "
        "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} eviction_count={} "
        "directory_entries={} page_bytes={} rows={} owned_pages={}",
        name,
        entry.fixed_width_pages_by_column.size(),
        fixed_width_page_count(entry),
        entry.fixed_width_page_metrics.resident_pages,
        entry.fixed_width_page_metrics.resident_bytes,
        entry.fixed_width_page_metrics.stats_pages,
        entry.fixed_width_page_metrics.evicted_pages,
        entry.fixed_width_page_metrics.eviction_count,
        entry.fixed_width_page_directory.size(),
        entry.fixed_width_page_size_bytes,
        entry.num_rows,
        own_page_storage ? 1 : 0);
      return;
    }
    // Same-row-count merge only applies when the completeness contracts match.
    // Mixing a full pin with a partial pin produces an entry whose columns came
    // from different row coverage — drop and rebuild instead.
    if (existing_it->second.num_rows == new_num_rows) {
      // Same-row-count merge MUST preserve per-chunk memory_space alignment
      // between existing and new entry. The round-robin counter restarts at
      // chunk 0 → GPU 0 per pin_table call, and chunks at index i across all
      // columns share a memory_space because they came from the same
      // chunked_parquet_reader::read_chunk() call. Two pin_table calls of the
      // same file_paths with the same chunk_read_limit MUST therefore produce
      // identical chunk_memory_spaces vectors. Reject any mismatch loudly
      // rather than silently aliasing.
      auto& entry = existing_it->second;
      if (entry.chunk_memory_spaces.size() != chunk_memory_spaces.size()) {
        throw std::runtime_error(
          "[sirius_scan_manager::insert_pinned_entry] merge mismatch — "
          "existing.chunk_memory_spaces.size() (" +
          std::to_string(entry.chunk_memory_spaces.size()) +
          ") != new chunk_memory_spaces.size() (" + std::to_string(chunk_memory_spaces.size()) +
          ")");
      }
      for (std::size_t i = 0; i < chunk_memory_spaces.size(); ++i) {
        if (entry.chunk_memory_spaces[i] != chunk_memory_spaces[i]) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] merge mismatch — "
            "chunk_memory_spaces[" +
            std::to_string(i) + "] differs between existing and new entry");
        }
      }
      // Same row count → merge unique columns into the existing entry.
      // Decide which column INDICES are new BEFORE iterating chunks. Doing
      // the contains() check per-chunk would let chunk 0 install a new
      // column and then chunks 1..N-1 see contains()==true and skip — leaving
      // the new column with only chunk 0 and tripping cached_split_provider's
      // "mismatched chunk count across requested columns" invariant.
      std::vector<bool> is_new_col(column_names.size(), false);
      for (std::size_t i = 0; i < column_names.size(); ++i) {
        is_new_col[i] = !entry.data_batches_by_column.contains(column_names[i]);
      }
      if (entry.fixed_width_page_size_bytes == 0) {
        entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();
      }
      bool const own_page_storage = fixed_page_owned_pages_enabled() && is_auto_fixed_page_entry(name);
      for (auto& table : data_tables) {
        if (!table) { continue; }
        auto cols = table->release();
        if (cols.size() != column_names.size()) {
          throw std::runtime_error(
            "[sirius_scan_manager::insert_pinned_entry] table column count " +
            std::to_string(cols.size()) + " does not match column_names size " +
            std::to_string(column_names.size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
          if (!is_new_col[i]) {
            // Column was already cached before this merge call — drop the
            // duplicate chunk.
            continue;
          }
          auto column            = std::move(cols[i]);
          auto& chunks           = entry.data_batches_by_column[std::string{column_names[i]}];
          auto const chunk_index = chunks.size();
          auto* chunk_space      = chunk_index < entry.chunk_memory_spaces.size()
                                     ? entry.chunk_memory_spaces[chunk_index]
                                     : nullptr;
          auto const page_file_path = entry.cache_info.resolved_file_paths.empty()
                                        ? std::string{}
                                        : entry.cache_info.resolved_file_paths.front();
          std::size_t chunk_global_row_offset = 0;
          if (own_page_storage) {
            auto pages_it = entry.fixed_width_pages_by_column.find(column_names[i]);
            if (pages_it != entry.fixed_width_pages_by_column.end()) {
              for (auto const& page : pages_it->second) {
                chunk_global_row_offset =
                  std::max(chunk_global_row_offset, page.global_row_offset + page.num_rows);
              }
            }
          } else {
            for (auto const& existing_chunk : chunks) {
              if (existing_chunk) {
                chunk_global_row_offset += static_cast<std::size_t>(existing_chunk->size());
              }
            }
          }
          index_fixed_width_column_pages(entry,
                                         name,
                                         page_file_path,
                                         column_names[i],
                                         column->view(),
                                         chunk_index,
                                         chunk_global_row_offset,
                                         chunk_space,
                                         entry.fixed_width_page_size_bytes,
                                         own_page_storage && is_fixed_width_page_candidate(column->view()));
          bool const fixed_candidate = is_fixed_width_page_candidate(column->view());
          if (own_page_storage && fixed_candidate) {
            chunks.emplace_back(nullptr);
          } else {
            chunks.emplace_back(std::move(column));
          }
        }
      }
      // Reflect the merged columns in cache_info so can_serve_with_columns'
      // superset match — and the gather it drives — actually see them. Append
      // only columns that received data above (an empty data_tables call must
      // not list a column with no backing chunks in data_batches_by_column).
      // column_ids and names grow together and we only append, so the projection
      // positions already handed out for existing columns stay valid.
      for (std::size_t i = 0; i < is_new_col.size(); ++i) {
        if (!is_new_col[i]) { continue; }
        if (!entry.data_batches_by_column.contains(column_names[i])) { continue; }
        entry.cache_info.column_ids.push_back(cache_info.column_ids[i]);
        entry.cache_info.names.push_back(column_names[i]);
      }
      entry.fixed_width_page_metrics = {};
      apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
      auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
      entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
      entry.fixed_width_page_metrics.eviction_count = eviction_count;
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_directory indexed table='{}' fixed_cols={} pages={} "
        "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} "
        "eviction_count={} directory_entries={} page_bytes={} rows={} owned_pages={}",
        name,
        entry.fixed_width_pages_by_column.size(),
        fixed_width_page_count(entry),
        entry.fixed_width_page_metrics.resident_pages,
        entry.fixed_width_page_metrics.resident_bytes,
        entry.fixed_width_page_metrics.stats_pages,
        entry.fixed_width_page_metrics.evicted_pages,
        entry.fixed_width_page_metrics.eviction_count,
        entry.fixed_width_page_directory.size(),
        entry.fixed_width_page_size_bytes,
        entry.num_rows,
        own_page_storage ? 1 : 0);
      return;
    }
    // Row count or completeness contract differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.cache_info          = std::move(cache_info);
  entry.chunk_memory_spaces = std::move(chunk_memory_spaces);
  entry.tier                = cucascade::memory::Tier::GPU;
  entry.num_rows            = new_num_rows;
  entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();
  bool const own_page_storage = fixed_page_owned_pages_enabled() && is_auto_fixed_page_entry(name);

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      auto column            = std::move(cols[i]);
      auto& chunks           = entry.data_batches_by_column[std::string{column_names[i]}];
      auto const chunk_index = chunks.size();
      auto* chunk_space      = chunk_index < entry.chunk_memory_spaces.size()
                                 ? entry.chunk_memory_spaces[chunk_index]
                                 : nullptr;
      auto const page_file_path = entry.cache_info.resolved_file_paths.empty()
                                    ? std::string{}
                                    : entry.cache_info.resolved_file_paths.front();
      std::size_t chunk_global_row_offset = 0;
      if (own_page_storage) {
        auto pages_it = entry.fixed_width_pages_by_column.find(column_names[i]);
        if (pages_it != entry.fixed_width_pages_by_column.end()) {
          for (auto const& page : pages_it->second) {
            chunk_global_row_offset =
              std::max(chunk_global_row_offset, page.global_row_offset + page.num_rows);
          }
        }
      } else {
        for (auto const& existing_chunk : chunks) {
          if (existing_chunk) {
            chunk_global_row_offset += static_cast<std::size_t>(existing_chunk->size());
          }
        }
      }
      index_fixed_width_column_pages(entry,
                                     name,
                                     page_file_path,
                                     column_names[i],
                                     column->view(),
                                     chunk_index,
                                     chunk_global_row_offset,
                                     chunk_space,
                                     entry.fixed_width_page_size_bytes,
                                     own_page_storage && is_fixed_width_page_candidate(column->view()));
      bool const fixed_candidate = is_fixed_width_page_candidate(column->view());
      if (own_page_storage && fixed_candidate) {
        chunks.emplace_back(nullptr);
      } else {
        chunks.emplace_back(std::move(column));
      }
    }
  }

  entry.fixed_width_page_metrics = {};
  _pinned_entries[name] = std::move(entry);
  auto& inserted_entry = _pinned_entries.at(name);
  apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
  auto eviction_count = inserted_entry.fixed_width_page_metrics.eviction_count;
  inserted_entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(inserted_entry);
  inserted_entry.fixed_width_page_metrics.eviction_count = eviction_count;
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] page_directory indexed table='{}' fixed_cols={} pages={} "
    "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} eviction_count={} "
    "directory_entries={} page_bytes={} rows={} owned_pages={}",
    name,
    inserted_entry.fixed_width_pages_by_column.size(),
    fixed_width_page_count(inserted_entry),
    inserted_entry.fixed_width_page_metrics.resident_pages,
    inserted_entry.fixed_width_page_metrics.resident_bytes,
    inserted_entry.fixed_width_page_metrics.stats_pages,
    inserted_entry.fixed_width_page_metrics.evicted_pages,
    inserted_entry.fixed_width_page_metrics.eviction_count,
    inserted_entry.fixed_width_page_directory.size(),
    inserted_entry.fixed_width_page_size_bytes,
    inserted_entry.num_rows,
    own_page_storage ? 1 : 0);
}

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  cache_entry_info cache_info,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space)
{
  // The host-tier path captures one chunk per emitted batch; each chunk holds every
  // pinned column. Re-insert always replaces — there is no per-column merge analog
  // to the GPU path because the chunk-vs-column dimensions are flipped.
  std::size_t new_num_rows = 0;
  for (auto const& chunk : host_chunks) {
    if (!chunk) { continue; }
    auto const& host_table = chunk->get_host_table();
    if (host_table && !host_table->columns.empty()) {
      new_num_rows += static_cast<std::size_t>(host_table->columns.front().num_rows);
    }
  }

  pinned_entry entry;
  entry.cache_info   = std::move(cache_info);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);

  {
    std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
    _pinned_entries[name] = std::move(entry);
  }
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
  _pinned_entries.erase(name);
  _fixed_page_admission_rejected_entries.erase(name);
}

void sirius_scan_manager::visit_pinned_entries(
  const std::function<bool(std::string_view, const pinned_entry&)>& visitor) const
{
  std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
  for (auto const& [name, entry] : _pinned_entries) {
    if (!visitor(name, entry)) { break; }
  }
}

bool sirius_scan_manager::try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op)
{
  const auto& table_info = op->get_ingestible().table_info();
  auto* parquet = dynamic_cast<op::scan::parquet_gpu_ingestible*>(&op->get_ingestible());
  if (parquet != nullptr && parquet->fixed_page_cache_has_dynamic_filters()) {
    SIRIUS_LOG_INFO("[fixed-page-cache] reuse_skip reason=dynamic_filter_scan operator='{}'",
                    op->get_operator_id());
    return false;
  }

  try {
    std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      // Identity + serviceability gate: empty when this cache cannot serve the scan
      // (wrong format / file-set / table, or missing a requested column).
      if (entry.cache_info.can_serve_with_columns(table_info).empty()) { continue; }
      // Serve cached columns in the ingestible's materialized (disk-decode) order rather
      // than raw column_ids order, so post_filter_and_project's index-based filter and
      // projection bind to the same columns they would on the disk read path.
      auto cols = gather_by_primary_index(entry.cache_info.column_ids,
                                          op->get_ingestible().materialized_column_order());
      if (cols.empty()) { continue; }  // defensive: materialized set must be a cache subset
      if (!entry.cache_info.filter_signature.empty()) {
        if (parquet == nullptr ||
            parquet->fixed_page_cache_filter_signature() != entry.cache_info.filter_signature) {
          continue;
        }
      }
      auto provider = make_provider_for_pinned_entry(entry, cols);
      if (!provider) { continue; }
      _metadata_processor->use_cached_entries_for_pipeline(op, std::move(provider));
      spdlog::info("[sirius_scan_manager] assigned pinned entry '{}' to operator '{}'",
                   pinned_name,
                   op->get_operator_id());
      return true;
    }
  } catch (...) {
    spdlog::error(
      "[sirius_scan_manager] error while trying to assign cached entries to "
      "operator '{}'",
      op->get_operator_id());
  }
  return false;
}

}  // namespace sirius::scan_manager
