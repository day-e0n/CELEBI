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

#include "scan_manager/gpu_ingestible_factory.hpp"

#include "data/sirius_converter_registry.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/utils.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/pinned_table_gpu_ingestible.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

namespace {

// wdy start
bool partial_pin_reuse_enabled()
{
  auto const* value = std::getenv("SIRIUS_ENABLE_PARTIAL_PIN_REUSE");
  return value != nullptr && std::string_view(value) == "1";
}

bool fixed_page_auto_cache_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE");
  return value != nullptr && std::string_view(value) == "1";
}

bool fixed_page_auto_cache_round_robin_chunks_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_ROUND_ROBIN_CHUNKS");
  return value == nullptr || std::string_view(value) != "0";
}

std::optional<std::size_t> parse_byte_size(std::string_view text)
{
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  if (text.empty()) { return std::nullopt; }

  std::size_t pos = 0;
  while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) { ++pos; }
  if (pos == 0) { return std::nullopt; }

  std::size_t value = 0;
  try {
    value = static_cast<std::size_t>(std::stoull(std::string{text.substr(0, pos)}));
  } catch (...) {
    return std::nullopt;
  }

  auto suffix = text.substr(pos);
  while (!suffix.empty() && std::isspace(static_cast<unsigned char>(suffix.front()))) {
    suffix.remove_prefix(1);
  }
  std::string normalized;
  normalized.reserve(suffix.size());
  for (auto ch : suffix) {
    normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }

  std::size_t multiplier = 1;
  if (normalized.empty() || normalized == "b") {
    multiplier = 1;
  } else if (normalized == "k" || normalized == "kb" || normalized == "kib") {
    multiplier = 1024ULL;
  } else if (normalized == "m" || normalized == "mb" || normalized == "mib") {
    multiplier = 1024ULL * 1024ULL;
  } else if (normalized == "g" || normalized == "gb" || normalized == "gib") {
    multiplier = 1024ULL * 1024ULL * 1024ULL;
  } else {
    return std::nullopt;
  }

  if (value > std::numeric_limits<std::size_t>::max() / multiplier) { return std::nullopt; }
  return value * multiplier;
}

std::size_t fixed_page_auto_cache_max_bytes()
{
  if (auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_MAX_BYTES")) {
    if (auto parsed = parse_byte_size(value)) { return *parsed; }
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] ignoring invalid SIRIUS_FIXED_PAGE_AUTO_CACHE_MAX_BYTES='{}'",
      value);
  }
  return 3ULL * 1024ULL * 1024ULL * 1024ULL;
}

std::string fixed_page_auto_cache_admission_policy()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_ADMISSION_POLICY");
  if (value == nullptr || *value == '\0') { return "smallest"; }
  std::string policy{value};
  if (policy == "smallest" || policy == "filter_order") { return policy; }
  SIRIUS_LOG_WARN(
    "[fixed-page-cache] ignoring invalid SIRIUS_FIXED_PAGE_AUTO_CACHE_ADMISSION_POLICY='{}'",
    policy);
  return "smallest";
}

std::size_t fixed_page_auto_cache_max_columns()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_MAX_COLUMNS");
  if (value == nullptr || *value == '\0') { return 1; }
  try {
    std::size_t pos = 0;
    auto parsed     = static_cast<std::size_t>(std::stoull(value, &pos));
    if (pos == std::string_view(value).size()) { return parsed; }
  } catch (...) {
  }
  SIRIUS_LOG_WARN(
    "[fixed-page-cache] ignoring invalid SIRIUS_FIXED_PAGE_AUTO_CACHE_MAX_COLUMNS='{}'",
    value);
  return 1;
}

double fixed_page_auto_cache_min_useful_ratio()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_MIN_USEFUL_RATIO");
  if (value == nullptr || *value == '\0') { return 0.20; }
  try {
    std::size_t pos = 0;
    auto parsed     = std::stod(value, &pos);
    if (pos == std::string_view(value).size() && parsed >= 0.0) { return parsed; }
  } catch (...) {
  }
  SIRIUS_LOG_WARN(
    "[fixed-page-cache] ignoring invalid SIRIUS_FIXED_PAGE_AUTO_CACHE_MIN_USEFUL_RATIO='{}'",
    value);
  return 0.20;
}

double fixed_page_auto_cache_expected_reuses()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_EXPECTED_REUSES");
  if (value == nullptr || *value == '\0') { return 2.0; }
  try {
    std::size_t pos = 0;
    auto parsed     = std::stod(value, &pos);
    if (pos == std::string_view(value).size() && parsed >= 0.0) { return parsed; }
  } catch (...) {
  }
  SIRIUS_LOG_WARN(
    "[fixed-page-cache] ignoring invalid SIRIUS_FIXED_PAGE_AUTO_CACHE_EXPECTED_REUSES='{}'",
    value);
  return 2.0;
}

double fixed_page_auto_cache_min_score()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE_MIN_SCORE");
  if (value == nullptr || *value == '\0') { return 0.25; }
  try {
    std::size_t pos = 0;
    auto parsed     = std::stod(value, &pos);
    if (pos == std::string_view(value).size() && parsed >= 0.0) { return parsed; }
  } catch (...) {
  }
  SIRIUS_LOG_WARN(
    "[fixed-page-cache] ignoring invalid SIRIUS_FIXED_PAGE_AUTO_CACHE_MIN_SCORE='{}'",
    value);
  return 0.25;
}

std::string join_column_names(std::vector<std::string> const& columns)
{
  std::string out;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    if (i != 0) { out += ","; }
    out += columns[i];
  }
  return out;
}

std::optional<std::size_t> estimate_parquet_rows(cudf::io::datasource& source)
{
  auto footer = cudf::io::parquet::fetch_footer_to_host(source);
  op::scan::hybrid_scan_reader reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                                      cudf::io::parquet_reader_options::builder().build());
  std::size_t rows = 0;
  for (auto const& row_group : reader.parquet_metadata().row_groups) {
    if (row_group.num_rows < 0) { return std::nullopt; }
    auto const rg_rows = static_cast<std::size_t>(row_group.num_rows);
    if (rows > std::numeric_limits<std::size_t>::max() - rg_rows) { return std::nullopt; }
    rows += rg_rows;
  }
  return rows;
}

std::optional<std::size_t> fixed_width_column_byte_size(
  op::scan::parquet_ingestible_table_info const& info,
  std::string const& column_name,
  std::size_t rows)
{
  for (std::size_t i = 0; i < info.names.size() && i < info.returned_types.size(); ++i) {
    if (info.names[i] != column_name) { continue; }
    auto const width = info.returned_types[i].fixed_width_byte_size();
    if (width == 0 || rows > std::numeric_limits<std::size_t>::max() / width) {
      return std::nullopt;
    }
    return rows * width;
  }
  return std::nullopt;
}

std::size_t sum_fixed_width_column_bytes(
  op::scan::parquet_ingestible_table_info const& info,
  std::vector<std::string> const& columns,
  std::size_t rows)
{
  std::size_t total = 0;
  for (auto const& column : columns) {
    auto bytes = fixed_width_column_byte_size(info, column, rows);
    if (!bytes) { return 0; }
    if (total > std::numeric_limits<std::size_t>::max() - *bytes) { return 0; }
    total += *bytes;
  }
  return total;
}

std::size_t sum_scan_fixed_width_bytes(op::scan::parquet_ingestible_table_info const& info,
                                       std::size_t rows)
{
  auto plan = op::scan::build_scan_plan(info.column_ids,
                                        info.projection_ids,
                                        info.names,
                                        info.returned_types,
                                        info.scan_output_arity,
                                        info.partition_indices);
  std::size_t total = 0;
  for (auto const& column : plan.data_columns) {
    if (column.primary_idx >= info.returned_types.size()) { continue; }
    auto const width = info.returned_types[column.primary_idx].fixed_width_byte_size();
    if (width == 0) { continue; }
    if (rows > std::numeric_limits<std::size_t>::max() / width) { return 0; }
    auto const bytes = rows * width;
    if (total > std::numeric_limits<std::size_t>::max() - bytes) { return 0; }
    total += bytes;
  }
  return total;
}

std::vector<std::string> choose_auto_cache_columns_within_budget(
  op::scan::parquet_ingestible_table_info const& info,
  std::vector<std::string> const& columns,
  std::size_t rows,
  std::size_t budget_bytes,
  std::string const& policy,
  std::size_t max_columns)
{
  struct candidate_column {
    std::string name;
    std::size_t bytes;
    std::size_t original_index;
  };

  std::vector<candidate_column> candidates;
  candidates.reserve(columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto bytes = fixed_width_column_byte_size(info, columns[i], rows);
    if (bytes) { candidates.push_back(candidate_column{columns[i], *bytes, i}); }
  }
  if (policy == "smallest") {
    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      if (lhs.bytes != rhs.bytes) { return lhs.bytes < rhs.bytes; }
      return lhs.original_index < rhs.original_index;
    });
  }

  std::size_t remaining = budget_bytes;
  std::vector<candidate_column> selected;
  for (auto const& candidate : candidates) {
    if (max_columns != 0 && selected.size() >= max_columns) { break; }
    if (candidate.bytes > remaining) { continue; }
    selected.push_back(candidate);
    remaining -= candidate.bytes;
  }
  std::sort(selected.begin(), selected.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.original_index < rhs.original_index;
  });

  std::vector<std::string> result;
  result.reserve(selected.size());
  for (auto const& column : selected) {
    result.push_back(column.name);
  }
  return result;
}

std::unordered_set<std::string>& failed_fixed_page_auto_cache_names()
{
  static std::unordered_set<std::string> names;
  return names;
}

std::mutex& failed_fixed_page_auto_cache_mutex()
{
  static std::mutex mutex;
  return mutex;
}

bool fixed_page_auto_cache_previously_failed(std::string const& cache_name)
{
  std::lock_guard<std::mutex> lock(failed_fixed_page_auto_cache_mutex());
  return failed_fixed_page_auto_cache_names().contains(cache_name);
}

void remember_fixed_page_auto_cache_failure(std::string const& cache_name)
{
  std::lock_guard<std::mutex> lock(failed_fixed_page_auto_cache_mutex());
  failed_fixed_page_auto_cache_names().insert(cache_name);
}

std::string strip_file_uri(std::string const& path)
{
  constexpr std::string_view kFile = "file://";
  if (path.size() > kFile.size() && path.substr(0, kFile.size()) == kFile) {
    return path.substr(kFile.size());
  }
  return path;
}

bool has_uri_scheme(std::string const& path) { return path.find("://") != std::string::npos; }

bool same_file_paths_exact(std::vector<std::string> const& lhs, std::vector<std::string> const& rhs)
{
  if (lhs.size() != rhs.size()) { return false; }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (strip_file_uri(lhs[i]) != strip_file_uri(rhs[i])) { return false; }
  }
  return true;
}

std::string auto_cache_name_for_paths(std::vector<std::string> const& paths)
{
  std::string path = paths.empty() ? std::string{"unknown"} : strip_file_uri(paths.front());
  auto const slash = path.find_last_of("/");
  auto base        = slash == std::string::npos ? path : path.substr(slash + 1);
  constexpr std::string_view parquet_ext = ".parquet";
  if (base.size() > parquet_ext.size() &&
      base.substr(base.size() - parquet_ext.size()) == parquet_ext) {
    base.resize(base.size() - parquet_ext.size());
  }
  return "__wdy_auto_fixed_page:" + base + ":" + std::to_string(std::hash<std::string>{}(path));
}

std::vector<cucascade::memory::memory_space*> sorted_gpu_spaces(
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces)
{
  std::vector<cucascade::memory::memory_space*> spaces;
  spaces.reserve(gpu_memory_spaces.size());
  for (auto const& [_, space] : gpu_memory_spaces) {
    if (space != nullptr) { spaces.push_back(space); }
  }
  std::sort(spaces.begin(), spaces.end(), [](auto* lhs, auto* rhs) {
    return lhs->get_device_id() < rhs->get_device_id();
  });
  return spaces;
}

std::vector<std::string> fixed_width_filter_columns_for_scan(
  op::scan::parquet_ingestible_table_info const& info)
{
  if (!info.table_filters || info.table_filters->filters.empty()) { return {}; }

  auto plan              = op::scan::build_scan_plan(info.column_ids,
                                        info.projection_ids,
                                        info.names,
                                        info.returned_types,
                                        info.scan_output_arity,
                                        info.partition_indices);
  auto duckdb_expression = op::convert_table_filters_to_expression(*info.table_filters,
                                                                   info.column_ids,
                                                                   info.returned_types,
                                                                   plan.batch_position_by_column_id,
                                                                   plan.partition_primary_indices);
  if (!duckdb_expression) { return {}; }

  auto ast = sirius::ast::from_duckdb(*duckdb_expression);
  if (!ast) { return {}; }

  std::unordered_set<std::size_t> seen;
  std::vector<std::size_t> refs;
  sirius::ast::visit_references(*ast, [&](sirius::ast::reference const& ref) {
    auto const data_idx = static_cast<std::size_t>(ref.column_index);
    if (seen.insert(data_idx).second) { refs.push_back(data_idx); }
  });
  std::sort(refs.begin(), refs.end());

  std::vector<std::string> columns;
  columns.reserve(refs.size());
  for (auto const data_idx : refs) {
    if (data_idx >= plan.data_columns.size()) { continue; }
    auto const& col = plan.data_columns[data_idx];
    if (col.primary_idx >= info.returned_types.size()) { continue; }
    if (col.name.empty() || !info.returned_types[col.primary_idx].is_fixed_width()) { continue; }
    columns.push_back(col.name);
  }
  return columns;
}

std::vector<std::string> missing_auto_cache_columns(
  std::unordered_map<std::string, pinned_entry> const& pinned_entries,
  std::string const& cache_name,
  std::vector<std::string> const& file_paths,
  std::vector<std::string> const& candidates)
{
  auto it = pinned_entries.find(cache_name);
  if (it == pinned_entries.end() || !same_file_paths_exact(it->second.file_paths, file_paths)) {
    return candidates;
  }

  // Do not extend an existing auto entry in this prototype. chunked_parquet_reader chunk
  // boundaries depend on the selected column byte width, and insert_pinned_entry requires
  // all columns at chunk index i to share identical row coverage and memory placement.
  return {};
}

bool try_auto_populate_fixed_page_cache(
  op::scan::parquet_ingestible_table_info const& info,
  sirius_scan_manager& mgr,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
  std::unordered_map<std::string, pinned_entry> const& pinned_entries,
  std::size_t op_id)
{
  if (!fixed_page_auto_cache_enabled()) { return false; }
  if (info.resolved_file_paths.size() != 1) {
    SIRIUS_LOG_DEBUG(
      "[fixed-page-cache] auto_populate skipped op_id={} reason=requires_single_file files={}",
      op_id,
      info.resolved_file_paths.size());
    return false;
  }

  auto const candidates = fixed_width_filter_columns_for_scan(info);
  if (candidates.empty()) {
    SIRIUS_LOG_DEBUG(
      "[fixed-page-cache] auto_populate skipped op_id={} reason=no_fixed_width_filter_columns",
      op_id);
    return false;
  }

  auto const cache_name = auto_cache_name_for_paths(info.resolved_file_paths);
  if (fixed_page_auto_cache_previously_failed(cache_name)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] auto_populate skipped name='{}' op_id={} reason=previous_failure",
      cache_name,
      op_id);
    return false;
  }

  auto columns_to_read =
    missing_auto_cache_columns(pinned_entries, cache_name, info.resolved_file_paths, candidates);
  if (columns_to_read.empty()) {
    SIRIUS_LOG_INFO("[fixed-page-cache] auto_populate_hit name='{}' op_id={} candidate_cols={}",
                    cache_name,
                    op_id,
                    candidates.size());
    return false;
  }

  auto gpu_spaces = sorted_gpu_spaces(gpu_memory_spaces);
  if (gpu_spaces.empty()) {
    SIRIUS_LOG_DEBUG("[fixed-page-cache] auto_populate skipped op_id={} reason=no_gpu_spaces",
                     op_id);
    return false;
  }

  auto const start = std::chrono::steady_clock::now();
  try {
    auto const& path = info.resolved_file_paths.front();
    auto datasource  = mgr.create_datasource(path);
    std::unique_ptr<cudf::io::datasource> fallback_source;
    cudf::io::datasource* source = datasource.get();
    if (source == nullptr) {
      auto const local_path = strip_file_uri(path);
      if (has_uri_scheme(local_path)) {
        SIRIUS_LOG_WARN(
          "[fixed-page-cache] auto_populate skipped op_id={} reason=no_datasource path={}",
          op_id,
          path);
        return false;
      }
      fallback_source = cudf::io::datasource::create(local_path);
      source          = fallback_source.get();
    }

    auto const max_cache_bytes = fixed_page_auto_cache_max_bytes();
    if (max_cache_bytes != 0) {
      try {
        auto const rows = estimate_parquet_rows(*source);
        if (rows) {
          auto const admission_policy = fixed_page_auto_cache_admission_policy();
          auto const max_columns      = fixed_page_auto_cache_max_columns();
          auto const estimated_bytes = sum_fixed_width_column_bytes(info, columns_to_read, *rows);
          auto const scan_fixed_width_bytes = sum_scan_fixed_width_bytes(info, *rows);
          auto const useful_ratio =
            scan_fixed_width_bytes == 0
              ? 0.0
              : static_cast<double>(estimated_bytes) / static_cast<double>(scan_fixed_width_bytes);
          auto const min_useful_ratio = fixed_page_auto_cache_min_useful_ratio();
          auto const expected_reuses  = fixed_page_auto_cache_expected_reuses();
          auto const min_cache_score  = fixed_page_auto_cache_min_score();
          if (useful_ratio < min_useful_ratio) {
            SIRIUS_LOG_INFO(
              "[fixed-page-cache] auto_populate skipped name='{}' op_id={} "
              "reason=low_useful_ratio estimated_bytes={} scan_fixed_width_bytes={} "
              "useful_ratio={} min_useful_ratio={} cols={} candidate_names='{}'",
              cache_name,
              op_id,
              estimated_bytes,
              scan_fixed_width_bytes,
              useful_ratio,
              min_useful_ratio,
              columns_to_read.size(),
              join_column_names(columns_to_read));
            return false;
          }
          if (estimated_bytes > max_cache_bytes) {
            auto selected = choose_auto_cache_columns_within_budget(
              info, columns_to_read, *rows, max_cache_bytes, admission_policy, max_columns);
            auto const selected_bytes = sum_fixed_width_column_bytes(info, selected, *rows);
            if (selected.empty()) {
              remember_fixed_page_auto_cache_failure(cache_name);
              SIRIUS_LOG_INFO(
                "[fixed-page-cache] auto_populate skipped name='{}' op_id={} "
                "reason=admission_estimate estimated_bytes={} max_bytes={} cols={}",
                cache_name,
                op_id,
                estimated_bytes,
                max_cache_bytes,
                columns_to_read.size());
              return false;
            }
            auto const selected_ratio =
              scan_fixed_width_bytes == 0
                ? 0.0
                : static_cast<double>(selected_bytes) /
                    static_cast<double>(scan_fixed_width_bytes);
            auto const cache_score = selected_ratio * expected_reuses;
            if (cache_score < min_cache_score) {
              SIRIUS_LOG_INFO(
                "[fixed-page-cache] auto_populate skipped name='{}' op_id={} "
                "reason=low_cache_score selected_bytes={} scan_fixed_width_bytes={} "
                "selected_ratio={} expected_reuses={} cache_score={} min_cache_score={} "
                "selected_names='{}'",
                cache_name,
                op_id,
                selected_bytes,
                scan_fixed_width_bytes,
                selected_ratio,
                expected_reuses,
                cache_score,
                min_cache_score,
                join_column_names(selected));
              return false;
            }
            SIRIUS_LOG_INFO(
              "[fixed-page-cache] auto_populate admission_trim name='{}' op_id={} "
              "estimated_bytes={} selected_bytes={} max_bytes={} cols={} selected_cols={} "
              "policy={} max_columns={} useful_ratio={} selected_ratio={} expected_reuses={} "
              "cache_score={} selected_names='{}'",
              cache_name,
              op_id,
              estimated_bytes,
              selected_bytes,
              max_cache_bytes,
              columns_to_read.size(),
              selected.size(),
              admission_policy,
              max_columns,
              useful_ratio,
              selected_ratio,
              expected_reuses,
              cache_score,
              join_column_names(selected));
            columns_to_read = std::move(selected);
          } else if (max_columns != 0 && columns_to_read.size() > max_columns) {
            auto selected = choose_auto_cache_columns_within_budget(
              info, columns_to_read, *rows, estimated_bytes, admission_policy, max_columns);
            auto const selected_bytes = sum_fixed_width_column_bytes(info, selected, *rows);
            auto const selected_ratio =
              scan_fixed_width_bytes == 0
                ? 0.0
                : static_cast<double>(selected_bytes) /
                    static_cast<double>(scan_fixed_width_bytes);
            auto const cache_score = selected_ratio * expected_reuses;
            if (cache_score < min_cache_score) {
              SIRIUS_LOG_INFO(
                "[fixed-page-cache] auto_populate skipped name='{}' op_id={} "
                "reason=low_cache_score selected_bytes={} scan_fixed_width_bytes={} "
                "selected_ratio={} expected_reuses={} cache_score={} min_cache_score={} "
                "selected_names='{}'",
                cache_name,
                op_id,
                selected_bytes,
                scan_fixed_width_bytes,
                selected_ratio,
                expected_reuses,
                cache_score,
                min_cache_score,
                join_column_names(selected));
              return false;
            }
            SIRIUS_LOG_INFO(
              "[fixed-page-cache] auto_populate admission_trim name='{}' op_id={} "
              "estimated_bytes={} selected_bytes={} max_bytes={} cols={} selected_cols={} "
              "policy={} max_columns={} useful_ratio={} selected_ratio={} expected_reuses={} "
              "cache_score={} selected_names='{}'",
              cache_name,
              op_id,
              estimated_bytes,
              selected_bytes,
              max_cache_bytes,
              columns_to_read.size(),
              selected.size(),
              admission_policy,
              max_columns,
              useful_ratio,
              selected_ratio,
              expected_reuses,
              cache_score,
              join_column_names(selected));
            columns_to_read = std::move(selected);
          }
        }
      } catch (std::exception const& e) {
        SIRIUS_LOG_DEBUG(
          "[fixed-page-cache] auto_populate admission estimate failed name='{}' op_id={} "
          "error={}",
          cache_name,
          op_id,
          e.what());
      }
    }

    auto* read_space = gpu_spaces.front();
    rmm::cuda_set_device_raii read_guard{rmm::cuda_device_id{read_space->get_device_id()}};

    auto file_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{source}).build();
    file_opts.set_column_names(columns_to_read);

    auto const chunk_read_limit = std::max<std::size_t>(info.approximate_batch_size, 1);
    cudf::io::chunked_parquet_reader reader(chunk_read_limit, file_opts);

    std::vector<std::unique_ptr<cudf::table>> tables;
    std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
    std::vector<std::string> read_column_names;
    std::size_t emitted_chunk_idx = 0;
    std::size_t total_rows        = 0;
    auto* registry = fixed_page_auto_cache_round_robin_chunks_enabled() && gpu_spaces.size() > 1
                       ? &sirius::converter_registry::get()
                       : nullptr;

    while (reader.has_next()) {
      auto chunk      = reader.read_chunk();
      auto chunk_rows = static_cast<std::size_t>(chunk.tbl->num_rows());
      if (chunk_rows == 0) { break; }
      if (read_column_names.empty()) {
        read_column_names.reserve(chunk.metadata.schema_info.size());
        for (auto const& col_info : chunk.metadata.schema_info) {
          read_column_names.push_back(col_info.name);
        }
      }

      auto* storage_space =
        registry != nullptr ? gpu_spaces[emitted_chunk_idx % gpu_spaces.size()] : read_space;
      if (storage_space != read_space) {
        cucascade::gpu_table_representation source_repr(
          std::move(chunk.tbl), *read_space, rmm::cuda_stream_default);
        auto moved_repr = registry->convert<cucascade::gpu_table_representation>(
          source_repr, storage_space, rmm::cuda_stream_default);
        rmm::cuda_set_device_raii storage_guard{
          rmm::cuda_device_id{storage_space->get_device_id()}};
        auto release_stream = storage_space->acquire_stream();
        tables.emplace_back(moved_repr->release_table(release_stream));
        release_stream.synchronize();
      } else {
        tables.emplace_back(std::move(chunk.tbl));
      }

      chunk_memory_spaces.push_back(storage_space);
      total_rows += chunk_rows;
      ++emitted_chunk_idx;
    }

    if (tables.empty() || read_column_names.empty()) {
      SIRIUS_LOG_DEBUG(
        "[fixed-page-cache] auto_populate skipped op_id={} reason=no_rows path={}", op_id, path);
      return false;
    }

    mgr.insert_pinned_entry(cache_name,
                            std::move(read_column_names),
                            info.resolved_file_paths,
                            std::move(tables),
                            std::move(chunk_memory_spaces),
                            false);

    auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] auto_populate name='{}' op_id={} cols={} chunks={} rows={} "
      "duration_us={} rr_chunks={} column_names='{}'",
      cache_name,
      op_id,
      columns_to_read.size(),
      emitted_chunk_idx,
      total_rows,
      duration_us,
      registry != nullptr ? 1 : 0,
      join_column_names(read_column_names));
    return true;
  } catch (std::exception const& e) {
    auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    remember_fixed_page_auto_cache_failure(cache_name);
    SIRIUS_LOG_WARN(
      "[fixed-page-cache] auto_populate failed name='{}' op_id={} cols={} duration_us={} "
      "error={}",
      cache_name,
      op_id,
      columns_to_read.size(),
      duration_us,
      e.what());
    return false;
  }
}
// wdy end

}  // namespace

gpu_ingestible_factory::gpu_ingestible_factory(
  std::unordered_map<std::string, pinned_entry> const& pinned_entries) noexcept
  : _pinned_entries(pinned_entries)
{
}

std::shared_ptr<io::gpu_ingestible> gpu_ingestible_factory::produce(
  std::unique_ptr<io::ingestible_table_info> table_info,
  sirius_scan_manager& mgr,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
  std::size_t op_id)
{
  if (!table_info) { return nullptr; }

  if (auto cached = try_cached(table_info, gpu_memory_spaces, op_id)) { return cached; }

  auto const* parquet_info =
    dynamic_cast<op::scan::parquet_ingestible_table_info const*>(table_info.get());
  auto ingestible = io::make_gpu_ingestible(std::move(table_info), mgr);

  // Populate after the current ingestible is constructed, so the first query does not
  // immediately consume the cache it just created. Reuse starts on the next scan/query.
  if (ingestible && parquet_info != nullptr) {
    try_auto_populate_fixed_page_cache(
      *parquet_info, mgr, gpu_memory_spaces, _pinned_entries, op_id);
  }

  return ingestible;
}

std::shared_ptr<io::gpu_ingestible> gpu_ingestible_factory::try_cached(
  std::unique_ptr<io::ingestible_table_info>& table_info,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces,
  std::size_t op_id) const
{
  if (!table_info) { return nullptr; }

  // Cache is parquet-only today (no pin_duckdb_table path). Cast probe;
  // non-parquet table_info falls through.
  auto const* parquet_info =
    dynamic_cast<op::scan::parquet_ingestible_table_info const*>(table_info.get());
  if (parquet_info == nullptr) { return nullptr; }
  auto const& info = *parquet_info;  // alias so the cache body reads unchanged

  // If a pinned entry's file paths match this table_info, build the same
  // scan_plan the parquet path would build and serve the scan from cache.
  auto matches_scan_info = [&info](const pinned_entry& entry) {
    if (entry.file_paths.size() != info.resolved_file_paths.size()) { return false; }
    auto sorted_a = entry.file_paths;
    auto sorted_b = info.resolved_file_paths;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    return sorted_a == sorted_b;
  };
  try {
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      if (!matches_scan_info(entry)) { continue; }
      // A partial pin (pin_table(..., n_rows=N) capped below the full file
      // content) MUST NOT serve cached reads by default — the incoming table_info
      // carries no n_rows budget, so a partial-entry hit would silently
      // mask missing rows. The fixed-page prototype enables this only via an
      // explicit experiment environment variable.
      if (entry.is_partial) {
        // wdy start
        if (!partial_pin_reuse_enabled()) {
          SIRIUS_LOG_DEBUG(
            "[gpu_ingestible_factory::try_cached] pinned entry '{}' matches op_id={} but is "
            "partial (row-count budget at pin time); falling through to per-format ingestible",
            pinned_name,
            op_id);
          break;
        }
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] partial_pin_reuse enabled pinned='{}' op_id={} cached_rows={} "
          "tier={}",
          pinned_name,
          op_id,
          entry.num_rows,
          entry.tier == cucascade::memory::Tier::GPU ? "gpu" : "host");
        // wdy end
      }

      // Build the canonical scan_plan once. Everything downstream — cached
      // column layout, filter pushdown indices, post-read assembly — reads
      // from this. Held by shared_ptr<const> so each emitted operator_data
      // can carry it to the gpu scan operator's per-task assembly check
      // without copying.
      auto plan_shared = std::make_shared<op::scan::scan_plan const>(
        op::scan::build_scan_plan(info.column_ids,
                                  info.projection_ids,
                                  info.names,
                                  info.returned_types,
                                  info.scan_output_arity,
                                  info.partition_indices));
      auto const& plan = *plan_shared;

      // Hive partitions on a cached scan would require per-chunk file_path
      // metadata that pinned entries don't carry today. Fall through to
      // the per-format path, which extracts partition values per file at
      // read time.
      if (plan.has_partitions()) {
        SIRIUS_LOG_DEBUG(
          "[gpu_ingestible_factory::try_cached] pinned entry '{}' matches op_id={} but scan "
          "has hive partitions; falling through to per-format ingestible",
          pinned_name,
          op_id);
        break;
      }

      // Filter expression: BoundReferences are in D-space, via
      // plan.batch_position_by_column_id. Same recipe parquet's run_batch
      // uses, so the filter evaluates correctly against the cached batch
      // (which is in D-order by construction above). Built before the
      // tier-specific assembly so both branches share the same filter.
      std::shared_ptr<duckdb::Expression> filter_expression;
      if (info.table_filters && !info.table_filters->filters.empty()) {
        auto duckdb_expression =
          op::convert_table_filters_to_expression(*info.table_filters,
                                                  info.column_ids,
                                                  info.returned_types,
                                                  plan.batch_position_by_column_id,
                                                  plan.partition_primary_indices);
        if (duckdb_expression) {
          filter_expression = std::shared_ptr<duckdb::Expression>(std::move(duckdb_expression));
        }
      }

      if (entry.tier == cucascade::memory::Tier::HOST) {
        // HOST-tier entries store one host_data_representation per chunk in
        // entry.host_chunks; chunk_memory_spaces is intentionally empty (see
        // pinned_entry doc comment + insert_pinned_entry_host). Validate the
        // host_chunks vector instead.
        if (entry.host_chunks.empty()) {
          throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned host entry '" +
                                   pinned_name + "' has no host_chunks");
        }
        for (std::size_t i = 0; i < entry.host_chunks.size(); ++i) {
          if (!entry.host_chunks[i]) {
            throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned host entry '" +
                                     pinned_name + "' host_chunks[" + std::to_string(i) +
                                     "] is null");
          }
        }
        // The HOST cached path materializes host chunks onto the executing
        // GPU via converter_registry.convert<gpu_table_representation>(...).
        // Without a GPU memory_space map there is no destination — fall
        // through to the per-format path so the query still succeeds.
        if (gpu_memory_spaces.empty()) {
          SIRIUS_LOG_DEBUG(
            "[gpu_ingestible_factory::try_cached] pinned host entry '{}' matches op_id={} "
            "but no gpu_memory_spaces map was provided; falling through to per-format "
            "ingestible",
            pinned_name,
            op_id);
          break;
        }

        // Map each D-position to its index inside the captured host chunk.
        // column_names is in capture order, so we look up the requested data
        // column by name. A missing column means the user pinned a subset
        // that doesn't cover this scan — fall back to the per-format path
        // so the query still succeeds.
        std::vector<std::size_t> column_indices;
        column_indices.reserve(plan.data_columns.size());
        for (auto const& dc : plan.data_columns) {
          auto it = std::find(entry.column_names.begin(), entry.column_names.end(), dc.name);
          if (it == entry.column_names.end()) {
            throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                     pinned_name + "' missing column '" + dc.name +
                                     "' required by scan op");
          }
          column_indices.push_back(
            static_cast<std::size_t>(std::distance(entry.column_names.begin(), it)));
        }

        SIRIUS_LOG_DEBUG(
          "[gpu_ingestible_factory::try_cached] using host pinned_table_gpu_ingestible for "
          "op_id={} (pinned='{}' data_cols={} chunks={} needs_assembly={})",
          op_id,
          pinned_name,
          column_indices.size(),
          entry.host_chunks.size(),
          op::scan::needs_output_assembly(plan));

        return std::make_shared<op::scan::pinned_table_gpu_ingestible>(std::move(table_info),
                                                                       entry.host_chunks,
                                                                       std::move(column_indices),
                                                                       *entry.memory_space,
                                                                       gpu_memory_spaces,
                                                                       std::move(filter_expression),
                                                                       std::move(plan_shared));
      }

      // GPU-tier validation: every cached chunk has an owning memory_space.
      // chunk_memory_spaces is parallel to the inner vectors of
      // data_batches_by_column; empty vector means no chunks; null entries
      // violate the chunks-at-index-i invariant.
      if (entry.chunk_memory_spaces.empty()) {
        throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                 pinned_name + "' has no chunk_memory_spaces");
      }
      for (std::size_t i = 0; i < entry.chunk_memory_spaces.size(); ++i) {
        if (entry.chunk_memory_spaces[i] == nullptr) {
          throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                   pinned_name + "' chunk_memory_spaces[" + std::to_string(i) +
                                   "] is null");
        }
      }

      // Look up the pinned chunks for each D-position by name. data_columns
      // is in D-order, so columns_per_request[d] is the chunk vector for
      // D-position d.
      std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request;
      columns_per_request.reserve(plan.data_columns.size());
      for (auto const& dc : plan.data_columns) {
        auto it = entry.data_batches_by_column.find(dc.name);
        if (it == entry.data_batches_by_column.end()) {
          throw std::runtime_error("[gpu_ingestible_factory::try_cached] pinned entry '" +
                                   pinned_name + "' missing column '" + dc.name +
                                   "' required by scan op");
        }
        columns_per_request.push_back(it->second);
      }

      SIRIUS_LOG_DEBUG(
        "[gpu_ingestible_factory::try_cached] using pinned_table_gpu_ingestible for op_id={} "
        "(pinned='{}' data_cols={} needs_assembly={})",
        op_id,
        pinned_name,
        columns_per_request.size(),
        op::scan::needs_output_assembly(plan));

      // Each chunk's data_batch is tagged with its actual memory_space so
      // data-locality scheduling fans cached-scan tasks across GPUs.
      return std::make_shared<op::scan::pinned_table_gpu_ingestible>(std::move(table_info),
                                                                     std::move(columns_per_request),
                                                                     entry.chunk_memory_spaces,
                                                                     std::move(filter_expression),
                                                                     std::move(plan_shared));
    }
  } catch (...) {
    SIRIUS_LOG_TRACE("not all the columns are pinned for this query");
  }
  return nullptr;
}

}  // namespace sirius::scan_manager
