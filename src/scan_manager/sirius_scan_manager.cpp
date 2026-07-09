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

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
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
  }

  std::shared_ptr<cucascade::data_batch> get_next_batch() override
  {
    auto index = _index.fetch_add(1);
    if (index >= _n_chunks) { return nullptr; }
    if (_entry.tier == cucascade::memory::Tier::GPU) {
      return get_device_databatch(index);
    } else if (_entry.tier == cucascade::memory::Tier::HOST) {
      return get_host_databatch(index);
    }
    return nullptr;
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
};

struct fixed_page_batch_range {
  std::size_t chunk_index{0};
  std::size_t row_offset{0};
  std::size_t num_rows{0};
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
  }

  [[nodiscard]] bool usable() const noexcept
  {
    return !_ranges.empty() && _covered_rows == _entry.num_rows;
  }

  [[nodiscard]] std::size_t batch_count() const noexcept { return _ranges.size(); }

  std::shared_ptr<cucascade::data_batch> get_next_batch() override
  {
    auto index = _index.fetch_add(1);
    if (index >= _ranges.size()) { return nullptr; }
    auto const& range = _ranges[index];
    if (!range.memory_space) { return nullptr; }

    if (range.memory_space->get_device_id() >= 0) {
      rmm::cuda_set_device_raii device_guard{
        rmm::cuda_device_id{range.memory_space->get_device_id()}};
      return get_device_databatch(range);
    }
    return get_device_databatch(range);
  }

 private:
  void build_ranges()
  {
    if (_entry.tier != cucascade::memory::Tier::GPU || _column_names.empty() ||
        _entry.fixed_width_pages_by_column.empty()) {
      return;
    }

    auto driver = choose_driver_column();
    if (!driver.has_value()) { return; }
    auto const& driver_pages = _entry.fixed_width_pages_by_column.at(_column_names[*driver]);
    for (auto const& page : driver_pages) {
      if (page.state != fixed_width_page_state::resident || page.num_rows == 0) { continue; }
      if (!all_columns_cover(page.chunk_index, page.row_offset, page.num_rows)) { continue; }
      _ranges.push_back(fixed_page_batch_range{page.chunk_index,
                                               page.row_offset,
                                               page.num_rows,
                                               page.memory_space});
      _covered_rows += page.num_rows;
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

  [[nodiscard]] bool all_columns_cover(std::size_t chunk_index,
                                       std::size_t row_offset,
                                       std::size_t num_rows) const
  {
    for (auto const& column_name : _column_names) {
      if (find_covering_fixed_page(_entry, column_name, chunk_index, row_offset, num_rows)) {
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

  std::shared_ptr<cucascade::data_batch> get_device_databatch(fixed_page_batch_range const& range)
  {
    std::vector<std::shared_ptr<cudf::column>> columns;
    std::vector<cudf::column_view> column_views;
    std::size_t alloc_size = 0;
    columns.reserve(_column_names.size());
    column_views.reserve(_column_names.size());

    for (auto const& column_name : _column_names) {
      auto const* page =
        find_covering_fixed_page(_entry, column_name, range.chunk_index, range.row_offset,
                                 range.num_rows);

      if (page) {
        if (auto owned = materialize_owned_subrange(*page, range.row_offset, range.num_rows)) {
          column_views.emplace_back(owned->view());
          alloc_size += owned->alloc_size();
          columns.push_back(std::move(owned));
          continue;
        }
      } else if (!fixed_page_hybrid_provider_enabled()) {
        return nullptr;
      }

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
      auto materialized = std::make_shared<cudf::column>(
        views.front(), cudf::get_default_stream(), cudf::get_current_device_resource_ref());
      column_views.emplace_back(materialized->view());
      alloc_size += materialized->alloc_size();
      columns.push_back(std::move(materialized));
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
                                    cudf::column const& column,
                                    std::size_t chunk_index,
                                    std::size_t chunk_global_row_offset,
                                    cucascade::memory::memory_space* memory_space,
                                    std::size_t page_size_bytes,
                                    bool own_page_storage)
{
  auto const view = column.view();
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

  std::unordered_map<int, std::size_t> resident_bytes_by_device;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : col_pages) {
      if (page.state != fixed_width_page_state::resident || page.key.device_id < 0) { continue; }
      resident_bytes_by_device[page.key.device_id] += page.num_bytes;
    }
  }

  struct eviction_candidate {
    fixed_width_column_page* page{nullptr};
    double score{0.0};
  };

  std::unordered_map<int, std::vector<eviction_candidate>> candidates_by_device;
  for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto& page : col_pages) {
      if (page.state != fixed_width_page_state::resident || page.key.device_id < 0) { continue; }
      auto const reuse_bonus = static_cast<double>(std::min<std::size_t>(page.access_count, 1024));
      candidates_by_device[page.key.device_id].push_back(
        eviction_candidate{&page, page.admission_score + reuse_bonus});
    }
  }

  std::size_t evicted_pages = 0;
  std::size_t evicted_bytes = 0;
  double min_evicted_score = std::numeric_limits<double>::infinity();
  double max_evicted_score = 0.0;
  for (auto& [device_id, candidates] : candidates_by_device) {
    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      if (lhs.score != rhs.score) { return lhs.score < rhs.score; }
      if (lhs.page->key.chunk_index != rhs.page->key.chunk_index) {
        return lhs.page->key.chunk_index < rhs.page->key.chunk_index;
      }
      return lhs.page->key.page_index < rhs.page->key.page_index;
    });
    auto& device_bytes = resident_bytes_by_device[device_id];
    for (auto const& candidate : candidates) {
      if (device_bytes <= budget) { break; }
      auto& page = *candidate.page;
      if (page.state != fixed_width_page_state::resident) { continue; }
      page.state = fixed_width_page_state::evicted;
      page.owned_column.reset();
      device_bytes -= std::min(device_bytes, page.num_bytes);
      ++entry.fixed_width_page_metrics.eviction_count;
      ++evicted_pages;
      evicted_bytes += page.num_bytes;
      min_evicted_score = std::min(min_evicted_score, candidate.score);
      max_evicted_score = std::max(max_evicted_score, candidate.score);
    }
  }

  if (evicted_pages != 0) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] page_budget applied table='{}' policy=score budget_bytes_per_gpu={} "
      "evicted_pages={} evicted_bytes={} min_score={} max_score={}",
      name,
      budget,
      evicted_pages,
      evicted_bytes,
      min_evicted_score,
      max_evicted_score);
  }
}

void apply_global_fixed_width_page_budget(std::unordered_map<std::string, pinned_entry>& entries)
{
  auto const budget = fixed_page_cache_budget_bytes_per_gpu();
  if (budget == 0) { return; }

  std::unordered_map<int, std::size_t> resident_bytes_by_device;
  struct eviction_candidate {
    pinned_entry* entry{nullptr};
    fixed_width_column_page* page{nullptr};
    std::string table_name;
    double score{0.0};
  };
  std::unordered_map<int, std::vector<eviction_candidate>> candidates_by_device;

  for (auto& [entry_name, entry] : entries) {
    for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
      for (auto& page : col_pages) {
        if (page.state != fixed_width_page_state::resident || page.key.device_id < 0) { continue; }
        resident_bytes_by_device[page.key.device_id] += page.num_bytes;
        auto const reuse_bonus = static_cast<double>(std::min<std::size_t>(page.access_count, 1024));
        candidates_by_device[page.key.device_id].push_back(
          eviction_candidate{&entry, &page, entry_name, page.admission_score + reuse_bonus});
      }
    }
  }

  for (auto& [device_id, candidates] : candidates_by_device) {
    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      if (lhs.score != rhs.score) { return lhs.score < rhs.score; }
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
    double min_evicted_score = std::numeric_limits<double>::infinity();
    double max_evicted_score = 0.0;
    for (auto const& candidate : candidates) {
      if (device_bytes <= budget) { break; }
      auto& page = *candidate.page;
      if (page.state != fixed_width_page_state::resident) { continue; }
      page.state = fixed_width_page_state::evicted;
      page.owned_column.reset();
      device_bytes -= std::min(device_bytes, page.num_bytes);
      ++candidate.entry->fixed_width_page_metrics.eviction_count;
      ++evicted_pages;
      evicted_bytes += page.num_bytes;
      min_evicted_score = std::min(min_evicted_score, candidate.score);
      max_evicted_score = std::max(max_evicted_score, candidate.score);
    }

    if (evicted_pages != 0) {
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_budget applied scope=global device={} policy=score "
        "budget_bytes_per_gpu={} evicted_pages={} evicted_bytes={} resident_bytes_after={} "
        "min_score={} max_score={}",
        device_id,
        budget,
        evicted_pages,
        evicted_bytes,
        device_bytes,
        min_evicted_score,
        max_evicted_score);
    }
  }

  for (auto& [_, entry] : entries) {
    auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
    entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
    entry.fixed_width_page_metrics.eviction_count = eviction_count;
  }
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
                                         *column,
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
      apply_global_fixed_width_page_budget(_pinned_entries);
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
                                         *column,
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
      apply_global_fixed_width_page_budget(_pinned_entries);
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
                                     *column,
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
  apply_global_fixed_width_page_budget(_pinned_entries);
  auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
  entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
  entry.fixed_width_page_metrics.eviction_count = eviction_count;
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] page_directory indexed table='{}' fixed_cols={} pages={} "
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

  _pinned_entries[name] = std::move(entry);
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
        auto* parquet = dynamic_cast<op::scan::parquet_gpu_ingestible*>(&op->get_ingestible());
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
