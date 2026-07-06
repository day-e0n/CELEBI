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

// sirius
#include <data/data_batch_utils.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/utils.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <io/io_context.hpp>
#include <io/prefetching_cache.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/parquet_metadata.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

// duckdb
#include <duckdb/common/hive_partitioning.hpp>

// uring_reactor MUST be included last among sirius headers — see
// parquet_split_provider.cpp for the BLOCK_SIZE macro-collision rationale.
#include <io/uring/uring_reactor.hpp>

// standard library
#include <algorithm>
// wdy start
#include <chrono>
// wdy end
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
// wdy start
#include <sstream>
// wdy end
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

struct rg_accumulator {
  std::vector<row_group_slice> slices;
  std::vector<parquet_split_info::fixed_page_row_range> fixed_page_row_ranges;
  std::size_t total_uncompressed_bytes = 0;
  // Partition values for the files currently bundled, in scan_plan::partition_columns order.
  // nullopt until the first file is added. Bundling is only safe across files with identical
  // values: post_filter_and_project synthesizes constant scalar columns from this single vector
  // on behalf of every file in the bundle, so all files in the bundle must share those values.
  std::optional<std::vector<std::string>> partition_values;
};

bool has_uri_scheme(std::string const& p) { return p.find("://") != std::string::npos; }

// Case-insensitively strip a leading "file://" so explicit local URIs behave
// exactly like bare paths (mirrors the scan_manager's normalize_path): the
// scheme check must not classify file:// as object-store, and cudf's bundled
// datasource wants a plain filesystem path.
std::string strip_file_uri(std::string const& p)
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

bool fixed_page_reuse_enabled()
{
  auto const* value = std::getenv("SIRIUS_ENABLE_FIXED_PAGE_REUSE");
  return value != nullptr && std::string_view(value) == "1";
}

bool same_file_paths_exact(std::vector<std::string> const& lhs, std::vector<std::string> const& rhs)
{
  if (lhs.size() != rhs.size()) { return false; }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (strip_file_uri(lhs[i]) != strip_file_uri(rhs[i])) { return false; }
  }
  return true;
}

std::vector<std::size_t> row_group_start_offsets(cudf::io::parquet::FileMetaData const& metadata)
{
  std::vector<std::size_t> offsets;
  offsets.reserve(metadata.row_groups.size());
  std::size_t row_offset = 0;
  for (auto const& row_group : metadata.row_groups) {
    offsets.push_back(row_offset);
    row_offset += static_cast<std::size_t>(row_group.num_rows);
  }
  return offsets;
}

std::size_t fixed_page_row_count(
  std::vector<parquet_split_info::fixed_page_row_range> const& row_ranges)
{
  std::size_t rows = 0;
  for (auto const& range : row_ranges) {
    rows += range.num_rows;
  }
  return rows;
}

std::size_t fixed_page_count_for_columns(scan_manager::pinned_entry const& entry,
                                         std::vector<std::size_t> const& cached_data_indices,
                                         scan_plan const& plan)
{
  std::size_t pages = 0;
  for (auto const data_idx : cached_data_indices) {
    auto const& name = plan.data_columns.at(data_idx).name;
    auto it          = entry.fixed_width_pages_by_column.find(name);
    if (it != entry.fixed_width_pages_by_column.end()) { pages += it->second.size(); }
  }
  return pages;
}

struct fixed_page_device_choice {
  int device_id{-1};
  std::size_t cached_bytes{0};
  std::size_t best_device_bytes{0};
  std::size_t device_count{0};
};

std::vector<std::size_t> cached_chunk_start_offsets(
  std::vector<std::shared_ptr<cudf::column>> const& chunks)
{
  std::vector<std::size_t> offsets;
  offsets.reserve(chunks.size() + 1);
  std::size_t row_offset = 0;
  offsets.push_back(row_offset);
  for (auto const& chunk : chunks) {
    if (!chunk) { throw std::runtime_error("[fixed-page-cache] null cached column chunk"); }
    row_offset += static_cast<std::size_t>(chunk->size());
    offsets.push_back(row_offset);
  }
  return offsets;
}

// wdy start
std::optional<std::size_t> cached_chunk_index_for_range(
  scan_manager::pinned_entry const& entry,
  std::vector<std::size_t> const& cached_data_indices,
  scan_plan const& plan,
  parquet_split_info::fixed_page_row_range const& range)
{
  if (cached_data_indices.empty()) { return std::nullopt; }

  auto const& column_name = plan.data_columns.at(cached_data_indices.front()).name;
  auto chunks_it          = entry.data_batches_by_column.find(column_name);
  if (chunks_it == entry.data_batches_by_column.end()) { return std::nullopt; }

  auto const range_end    = range.row_offset + range.num_rows;
  std::size_t chunk_start = 0;
  for (std::size_t chunk_index = 0; chunk_index < chunks_it->second.size(); ++chunk_index) {
    auto const& chunk = chunks_it->second[chunk_index];
    if (!chunk) { throw std::runtime_error("[fixed-page-cache] null cached column chunk"); }
    auto const chunk_end = chunk_start + static_cast<std::size_t>(chunk->size());
    if (range.row_offset >= chunk_start && range_end <= chunk_end) { return chunk_index; }
    chunk_start = chunk_end;
  }
  return std::nullopt;
}

bool fixed_page_view_aligned_splits_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_VIEW_ALIGNED_SPLITS");
  return value != nullptr && std::string_view(value) == "1";
}

bool fixed_page_zero_copy_output_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_ZERO_COPY_OUTPUT");
  return value == nullptr || std::string_view(value) != "0";
}

bool fixed_page_filtered_reuse_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_FILTERED_REUSE");
  return value != nullptr && std::string_view(value) == "1";
}

std::size_t fixed_page_min_filtered_split_cached_bytes()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_MIN_FILTERED_SPLIT_CACHED_BYTES");
  if (value == nullptr || *value == '\0') { return 0; }
  char* end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (end == value) { return 0; }
  return static_cast<std::size_t>(parsed);
}

bool fixed_page_pruning_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_PRUNING");
  return value != nullptr && std::string_view(value) == "1";
}

bool fixed_page_pruning_skip_partial_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_PRUNING_SKIP_PARTIAL");
  return value == nullptr || std::string_view(value) != "0";
}

struct fixed_page_pruning_literal {
  scan_manager::fixed_width_page_stat_kind kind{scan_manager::fixed_width_page_stat_kind::none};
  int64_t signed_value{0};
  uint64_t unsigned_value{0};
  double floating_value{0.0};
};

struct fixed_page_pruning_constraint {
  bool has_lower{false};
  fixed_page_pruning_literal lower;
  bool lower_inclusive{true};
  bool has_upper{false};
  fixed_page_pruning_literal upper;
  bool upper_inclusive{true};
};

struct fixed_page_pruning_summary {
  std::size_t matched_columns{0};
  std::size_t pages_considered{0};
  std::size_t all_fail_pages{0};
  std::size_t all_pass_pages{0};
  std::size_t partial_pages{0};
  std::size_t unknown_pages{0};
};

std::size_t row_range_overlap(std::size_t lhs_offset,
                              std::size_t lhs_rows,
                              std::size_t rhs_offset,
                              std::size_t rhs_rows);

using fixed_page_pruning_constraints =
  std::unordered_map<std::size_t, fixed_page_pruning_constraint>;

std::vector<std::size_t> filter_reference_data_indices(sirius::ast::node const& root)
{
  std::vector<std::size_t> refs;
  std::unordered_set<std::size_t> seen;
  sirius::ast::visit_references(root, [&](sirius::ast::reference const& ref) {
    auto const data_idx = static_cast<std::size_t>(ref.column_index);
    if (seen.insert(data_idx).second) { refs.push_back(data_idx); }
  });
  std::sort(refs.begin(), refs.end());
  return refs;
}

bool filter_references_are_cached(std::vector<std::size_t> const& refs,
                                  std::vector<bool> const& cached_by_data_index)
{
  if (refs.empty()) { return false; }
  for (auto const data_idx : refs) {
    if (data_idx >= cached_by_data_index.size() || !cached_by_data_index[data_idx]) {
      return false;
    }
  }
  return true;
}

template <class Xform>
std::vector<std::unique_ptr<sirius::ast::node>> remap_filter_children(
  std::vector<std::unique_ptr<sirius::ast::node>> const& children, Xform const& xform)
{
  std::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(children.size());
  for (auto const& child : children) {
    out.push_back(xform(child));
  }
  return out;
}

std::unique_ptr<sirius::ast::node> remap_filter_references(
  sirius::ast::node const& src, std::unordered_map<std::size_t, std::size_t> const& remap)
{
  auto xform = [&](std::unique_ptr<sirius::ast::node> const& child)
    -> std::unique_ptr<sirius::ast::node> {
    return child ? remap_filter_references(*child, remap) : nullptr;
  };

  return std::visit(
    [&](auto const& alt) -> std::unique_ptr<sirius::ast::node> {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, sirius::ast::reference>) {
        auto const data_idx = static_cast<std::size_t>(alt.column_index);
        auto const it       = remap.find(data_idx);
        if (it == remap.end()) {
          throw std::runtime_error(
            "[fixed-page-cache] filter references a column that is not in the cached filter table");
        }
        if (it->second > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
          throw std::overflow_error("[fixed-page-cache] remapped filter reference exceeds uint32");
        }
        return std::make_unique<sirius::ast::node>(sirius::ast::reference{
          static_cast<uint32_t>(it->second), alt.return_type});
      } else if constexpr (std::is_same_v<T, sirius::ast::constant>) {
        return std::make_unique<sirius::ast::node>(
          sirius::ast::constant{alt.payload, alt.return_type});
      } else if constexpr (std::is_same_v<T, sirius::ast::comparison>) {
        return std::make_unique<sirius::ast::node>(
          sirius::ast::comparison{alt.op, xform(alt.left), xform(alt.right)});
      } else if constexpr (std::is_same_v<T, sirius::ast::conjunction>) {
        return std::make_unique<sirius::ast::node>(sirius::ast::conjunction{
          alt.op, remap_filter_children(alt.children, xform)});
      } else if constexpr (std::is_same_v<T, sirius::ast::between>) {
        return std::make_unique<sirius::ast::node>(sirius::ast::between{xform(alt.input),
                                                                       xform(alt.lower),
                                                                       xform(alt.upper),
                                                                       alt.lower_inclusive,
                                                                       alt.upper_inclusive});
      } else if constexpr (std::is_same_v<T, sirius::ast::case_expr>) {
        std::vector<sirius::ast::case_expr::when_then> cases;
        cases.reserve(alt.cases.size());
        for (auto const& wt : alt.cases) {
          cases.push_back(sirius::ast::case_expr::when_then{xform(wt.when_), xform(wt.then_)});
        }
        return std::make_unique<sirius::ast::node>(
          sirius::ast::case_expr{std::move(cases), xform(alt.else_), alt.return_type()});
      } else if constexpr (std::is_same_v<T, sirius::ast::cast>) {
        return std::make_unique<sirius::ast::node>(
          sirius::ast::cast{xform(alt.child), alt.target_type, alt.try_cast});
      } else if constexpr (std::is_same_v<T, sirius::ast::unary_op>) {
        return std::make_unique<sirius::ast::node>(
          sirius::ast::unary_op{alt.op, xform(alt.child)});
      } else if constexpr (std::is_same_v<T, sirius::ast::coalesce>) {
        return std::make_unique<sirius::ast::node>(sirius::ast::coalesce{
          remap_filter_children(alt.children, xform), alt.return_type()});
      } else if constexpr (std::is_same_v<T, sirius::ast::in_list>) {
        return std::make_unique<sirius::ast::node>(sirius::ast::in_list{
          xform(alt.probe), remap_filter_children(alt.values, xform), alt.negated});
      } else if constexpr (std::is_same_v<T, sirius::ast::function_call>) {
        return std::make_unique<sirius::ast::node>(sirius::ast::function_call{
          alt.function(), remap_filter_children(alt.arguments(), xform), alt.return_type()});
      } else if constexpr (std::is_same_v<T, sirius::ast::aggregate>) {
        return std::make_unique<sirius::ast::node>(sirius::ast::aggregate{
          alt.function(),
          remap_filter_children(alt.arguments(), xform),
          alt.return_type(),
          alt.distinct()});
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in remap_filter_references");
      }
    },
    src.v);
}

std::optional<long double> pruning_literal_value(fixed_page_pruning_literal const& literal)
{
  switch (literal.kind) {
    case scan_manager::fixed_width_page_stat_kind::signed_int:
      return static_cast<long double>(literal.signed_value);
    case scan_manager::fixed_width_page_stat_kind::unsigned_int:
      return static_cast<long double>(literal.unsigned_value);
    case scan_manager::fixed_width_page_stat_kind::floating:
      return static_cast<long double>(literal.floating_value);
    default: return std::nullopt;
  }
}

std::optional<long double> pruning_stats_min(
  scan_manager::fixed_width_page_stats const& stats)
{
  if (!stats.valid) { return std::nullopt; }
  switch (stats.kind) {
    case scan_manager::fixed_width_page_stat_kind::signed_int:
      return static_cast<long double>(stats.min_signed);
    case scan_manager::fixed_width_page_stat_kind::unsigned_int:
      return static_cast<long double>(stats.min_unsigned);
    case scan_manager::fixed_width_page_stat_kind::floating:
      return static_cast<long double>(stats.min_floating);
    default: return std::nullopt;
  }
}

std::optional<long double> pruning_stats_max(
  scan_manager::fixed_width_page_stats const& stats)
{
  if (!stats.valid) { return std::nullopt; }
  switch (stats.kind) {
    case scan_manager::fixed_width_page_stat_kind::signed_int:
      return static_cast<long double>(stats.max_signed);
    case scan_manager::fixed_width_page_stat_kind::unsigned_int:
      return static_cast<long double>(stats.max_unsigned);
    case scan_manager::fixed_width_page_stat_kind::floating:
      return static_cast<long double>(stats.max_floating);
    default: return std::nullopt;
  }
}

bool pruning_literal_from_value(sirius::value const& value, fixed_page_pruning_literal& out)
{
  if (auto v = std::get_if<int8_t>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = *v;
    return true;
  }
  if (auto v = std::get_if<int16_t>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = *v;
    return true;
  }
  if (auto v = std::get_if<int32_t>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = *v;
    return true;
  }
  if (auto v = std::get_if<int64_t>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = *v;
    return true;
  }
  if (auto v = std::get_if<uint8_t>(&value)) {
    out.kind           = scan_manager::fixed_width_page_stat_kind::unsigned_int;
    out.unsigned_value = *v;
    return true;
  }
  if (auto v = std::get_if<uint16_t>(&value)) {
    out.kind           = scan_manager::fixed_width_page_stat_kind::unsigned_int;
    out.unsigned_value = *v;
    return true;
  }
  if (auto v = std::get_if<uint32_t>(&value)) {
    out.kind           = scan_manager::fixed_width_page_stat_kind::unsigned_int;
    out.unsigned_value = *v;
    return true;
  }
  if (auto v = std::get_if<uint64_t>(&value)) {
    out.kind           = scan_manager::fixed_width_page_stat_kind::unsigned_int;
    out.unsigned_value = *v;
    return true;
  }
  if (auto v = std::get_if<float>(&value)) {
    out.kind           = scan_manager::fixed_width_page_stat_kind::floating;
    out.floating_value = *v;
    return true;
  }
  if (auto v = std::get_if<double>(&value)) {
    out.kind           = scan_manager::fixed_width_page_stat_kind::floating;
    out.floating_value = *v;
    return true;
  }
  if (auto v = std::get_if<sirius::date_value>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = v->days;
    return true;
  }
  if (auto v = std::get_if<sirius::timestamp_sec_value>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = v->value;
    return true;
  }
  if (auto v = std::get_if<sirius::timestamp_ms_value>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = v->value;
    return true;
  }
  if (auto v = std::get_if<sirius::timestamp_us_value>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = v->value;
    return true;
  }
  if (auto v = std::get_if<sirius::timestamp_ns_value>(&value)) {
    out.kind         = scan_manager::fixed_width_page_stat_kind::signed_int;
    out.signed_value = v->value;
    return true;
  }
  return false;
}

void merge_lower_bound(fixed_page_pruning_constraint& constraint,
                       fixed_page_pruning_literal literal,
                       bool inclusive)
{
  if (!constraint.has_lower) {
    constraint.has_lower       = true;
    constraint.lower           = literal;
    constraint.lower_inclusive = inclusive;
    return;
  }
  auto current = pruning_literal_value(constraint.lower);
  auto next    = pruning_literal_value(literal);
  if (!current || !next) { return; }
  if (*next > *current || (*next == *current && constraint.lower_inclusive && !inclusive)) {
    constraint.lower           = literal;
    constraint.lower_inclusive = inclusive;
  }
}

void merge_upper_bound(fixed_page_pruning_constraint& constraint,
                       fixed_page_pruning_literal literal,
                       bool inclusive)
{
  if (!constraint.has_upper) {
    constraint.has_upper       = true;
    constraint.upper           = literal;
    constraint.upper_inclusive = inclusive;
    return;
  }
  auto current = pruning_literal_value(constraint.upper);
  auto next    = pruning_literal_value(literal);
  if (!current || !next) { return; }
  if (*next < *current || (*next == *current && constraint.upper_inclusive && !inclusive)) {
    constraint.upper           = literal;
    constraint.upper_inclusive = inclusive;
  }
}

sirius::comparison_type flip_comparison(sirius::comparison_type op)
{
  switch (op) {
    case sirius::comparison_type::lt: return sirius::comparison_type::gt;
    case sirius::comparison_type::le: return sirius::comparison_type::ge;
    case sirius::comparison_type::gt: return sirius::comparison_type::lt;
    case sirius::comparison_type::ge: return sirius::comparison_type::le;
    default: return op;
  }
}

bool add_comparison_constraint(fixed_page_pruning_constraints& constraints,
                               std::size_t data_idx,
                               sirius::comparison_type op,
                               fixed_page_pruning_literal literal)
{
  auto& constraint = constraints[data_idx];
  switch (op) {
    case sirius::comparison_type::equal:
      merge_lower_bound(constraint, literal, true);
      merge_upper_bound(constraint, literal, true);
      return true;
    case sirius::comparison_type::lt:
      merge_upper_bound(constraint, literal, false);
      return true;
    case sirius::comparison_type::le:
      merge_upper_bound(constraint, literal, true);
      return true;
    case sirius::comparison_type::gt:
      merge_lower_bound(constraint, literal, false);
      return true;
    case sirius::comparison_type::ge:
      merge_lower_bound(constraint, literal, true);
      return true;
    default: return false;
  }
}

bool extract_fixed_page_pruning_constraints(sirius::ast::node const& node,
                                            fixed_page_pruning_constraints& constraints,
                                            scan_plan const& plan);

bool extract_comparison_pruning_constraint(sirius::ast::comparison const& comparison,
                                           fixed_page_pruning_constraints& constraints,
                                           scan_plan const& plan)
{
  if (!comparison.left || !comparison.right) { return false; }

  auto const* left_ref = std::get_if<sirius::ast::reference>(&comparison.left->v);
  auto const* right_ref = std::get_if<sirius::ast::reference>(&comparison.right->v);
  auto const* left_const = std::get_if<sirius::ast::constant>(&comparison.left->v);
  auto const* right_const = std::get_if<sirius::ast::constant>(&comparison.right->v);

  sirius::ast::reference const* ref      = nullptr;
  sirius::ast::constant const* constant  = nullptr;
  auto op                                = comparison.op;
  if (left_ref != nullptr && right_const != nullptr) {
    ref      = left_ref;
    constant = right_const;
  } else if (left_const != nullptr && right_ref != nullptr) {
    ref      = right_ref;
    constant = left_const;
    op       = flip_comparison(op);
  } else {
    return false;
  }

  auto const data_idx = static_cast<std::size_t>(ref->column_index);
  if (data_idx >= plan.data_columns.size()) { return false; }

  fixed_page_pruning_literal literal;
  if (!pruning_literal_from_value(constant->payload, literal)) { return false; }
  return add_comparison_constraint(constraints, data_idx, op, literal);
}

bool extract_between_pruning_constraint(sirius::ast::between const& between,
                                        fixed_page_pruning_constraints& constraints,
                                        scan_plan const& plan)
{
  if (!between.input || !between.lower || !between.upper) { return false; }
  auto const* ref = std::get_if<sirius::ast::reference>(&between.input->v);
  auto const* lower_const = std::get_if<sirius::ast::constant>(&between.lower->v);
  auto const* upper_const = std::get_if<sirius::ast::constant>(&between.upper->v);
  if (ref == nullptr || lower_const == nullptr || upper_const == nullptr) { return false; }

  auto const data_idx = static_cast<std::size_t>(ref->column_index);
  if (data_idx >= plan.data_columns.size()) { return false; }

  fixed_page_pruning_literal lower;
  fixed_page_pruning_literal upper;
  if (!pruning_literal_from_value(lower_const->payload, lower) ||
      !pruning_literal_from_value(upper_const->payload, upper)) {
    return false;
  }

  auto& constraint = constraints[data_idx];
  merge_lower_bound(constraint, lower, between.lower_inclusive);
  merge_upper_bound(constraint, upper, between.upper_inclusive);
  return true;
}

bool extract_fixed_page_pruning_constraints(sirius::ast::node const& node,
                                            fixed_page_pruning_constraints& constraints,
                                            scan_plan const& plan)
{
  if (auto const* comparison = std::get_if<sirius::ast::comparison>(&node.v)) {
    return extract_comparison_pruning_constraint(*comparison, constraints, plan);
  }
  if (auto const* between = std::get_if<sirius::ast::between>(&node.v)) {
    return extract_between_pruning_constraint(*between, constraints, plan);
  }
  if (auto const* conjunction = std::get_if<sirius::ast::conjunction>(&node.v)) {
    if (conjunction->op != sirius::ast::conjunction::kind::op_and) { return false; }
    bool extracted_any = false;
    for (auto const& child : conjunction->children) {
      if (child && extract_fixed_page_pruning_constraints(*child, constraints, plan)) {
        extracted_any = true;
      }
    }
    return extracted_any;
  }
  return false;
}

enum class fixed_page_pruning_page_class : uint8_t { unknown, all_fail, all_pass, partial };

fixed_page_pruning_page_class classify_fixed_page_for_constraint(
  scan_manager::fixed_width_page_stats const& stats,
  fixed_page_pruning_constraint const& constraint)
{
  auto min_value = pruning_stats_min(stats);
  auto max_value = pruning_stats_max(stats);
  if (!min_value || !max_value) { return fixed_page_pruning_page_class::unknown; }

  bool all_pass = true;
  if (constraint.has_lower) {
    auto lower = pruning_literal_value(constraint.lower);
    if (!lower) { return fixed_page_pruning_page_class::unknown; }
    bool const lower_all_fail =
      constraint.lower_inclusive ? (*max_value < *lower) : (*max_value <= *lower);
    if (lower_all_fail) { return fixed_page_pruning_page_class::all_fail; }
    bool const lower_all_pass =
      constraint.lower_inclusive ? (*min_value >= *lower) : (*min_value > *lower);
    all_pass = all_pass && lower_all_pass;
  }

  if (constraint.has_upper) {
    auto upper = pruning_literal_value(constraint.upper);
    if (!upper) { return fixed_page_pruning_page_class::unknown; }
    bool const upper_all_fail =
      constraint.upper_inclusive ? (*min_value > *upper) : (*min_value >= *upper);
    if (upper_all_fail) { return fixed_page_pruning_page_class::all_fail; }
    bool const upper_all_pass =
      constraint.upper_inclusive ? (*max_value <= *upper) : (*max_value < *upper);
    all_pass = all_pass && upper_all_pass;
  }

  if (all_pass && !stats.has_null) { return fixed_page_pruning_page_class::all_pass; }
  return fixed_page_pruning_page_class::partial;
}

fixed_page_pruning_summary summarize_fixed_page_pruning(
  scan_manager::pinned_entry const& entry,
  std::vector<std::size_t> const& cached_data_indices,
  scan_plan const& plan,
  std::vector<parquet_split_info::fixed_page_row_range> const& row_ranges,
  fixed_page_pruning_constraints const& constraints)
{
  fixed_page_pruning_summary summary;
  if (constraints.empty()) { return summary; }

  for (auto const data_idx : cached_data_indices) {
    auto constraint_it = constraints.find(data_idx);
    if (constraint_it == constraints.end()) { continue; }

    auto const& column_name = plan.data_columns.at(data_idx).name;
    auto pages_it           = entry.fixed_width_pages_by_column.find(column_name);
    auto chunks_it          = entry.data_batches_by_column.find(column_name);
    if (pages_it == entry.fixed_width_pages_by_column.end() ||
        chunks_it == entry.data_batches_by_column.end()) {
      continue;
    }

    ++summary.matched_columns;
    auto const chunk_offsets = cached_chunk_start_offsets(chunks_it->second);
    for (auto const& page : pages_it->second) {
      if (page.chunk_index + 1 >= chunk_offsets.size()) { continue; }
      auto const page_row_offset = chunk_offsets[page.chunk_index] + page.row_offset;
      bool overlaps_split        = false;
      for (auto const& range : row_ranges) {
        if (row_range_overlap(page_row_offset, page.num_rows, range.row_offset, range.num_rows) !=
            0) {
          overlaps_split = true;
          break;
        }
      }
      if (!overlaps_split) { continue; }

      ++summary.pages_considered;
      switch (classify_fixed_page_for_constraint(page.stats, constraint_it->second)) {
        case fixed_page_pruning_page_class::all_fail: ++summary.all_fail_pages; break;
        case fixed_page_pruning_page_class::all_pass: ++summary.all_pass_pages; break;
        case fixed_page_pruning_page_class::partial: ++summary.partial_pages; break;
        case fixed_page_pruning_page_class::unknown: ++summary.unknown_pages; break;
      }
    }
  }

  return summary;
}

// wdy end

std::size_t row_range_overlap(std::size_t lhs_offset,
                              std::size_t lhs_rows,
                              std::size_t rhs_offset,
                              std::size_t rhs_rows)
{
  auto const lhs_end = lhs_offset + lhs_rows;
  auto const rhs_end = rhs_offset + rhs_rows;
  if (lhs_offset >= rhs_end || rhs_offset >= lhs_end) { return 0; }
  auto const begin = std::max(lhs_offset, rhs_offset);
  auto const end   = std::min(lhs_end, rhs_end);
  return end - begin;
}

fixed_page_device_choice choose_fixed_page_device(
  scan_manager::pinned_entry const& entry,
  std::vector<std::size_t> const& cached_data_indices,
  scan_plan const& plan,
  std::vector<parquet_split_info::fixed_page_row_range> const& row_ranges)
{
  std::unordered_map<int, std::size_t> bytes_by_device;
  std::size_t total_cached_bytes = 0;

  for (auto const data_idx : cached_data_indices) {
    auto const& column_name = plan.data_columns.at(data_idx).name;
    auto pages_it           = entry.fixed_width_pages_by_column.find(column_name);
    auto chunks_it          = entry.data_batches_by_column.find(column_name);
    if (pages_it == entry.fixed_width_pages_by_column.end() ||
        chunks_it == entry.data_batches_by_column.end()) {
      continue;
    }

    auto const chunk_offsets = cached_chunk_start_offsets(chunks_it->second);
    for (auto const& page : pages_it->second) {
      if (page.memory_space == nullptr || page.chunk_index + 1 >= chunk_offsets.size()) {
        continue;
      }
      auto const page_row_offset = chunk_offsets[page.chunk_index] + page.row_offset;
      for (auto const& range : row_ranges) {
        auto const overlap_rows =
          row_range_overlap(page_row_offset, page.num_rows, range.row_offset, range.num_rows);
        if (overlap_rows == 0) { continue; }
        auto const bytes = overlap_rows * page.element_size_bytes;
        bytes_by_device[page.memory_space->get_device_id()] += bytes;
        total_cached_bytes += bytes;
      }
    }
  }

  fixed_page_device_choice choice;
  choice.cached_bytes = total_cached_bytes;
  choice.device_count = bytes_by_device.size();
  for (auto const& [device_id, bytes] : bytes_by_device) {
    if (choice.device_id < 0 || bytes > choice.best_device_bytes ||
        (bytes == choice.best_device_bytes && device_id < choice.device_id)) {
      choice.device_id         = device_id;
      choice.best_device_bytes = bytes;
    }
  }
  return choice;
}

std::size_t fixed_page_candidate_device_count(scan_manager::pinned_entry const& entry)
{
  std::unordered_set<int> device_ids;
  for (auto* space : entry.chunk_memory_spaces) {
    if (space != nullptr) { device_ids.insert(space->get_device_id()); }
  }
  return device_ids.size();
}

std::unique_ptr<cudf::column> materialize_cached_fixed_column(
  scan_manager::pinned_entry const& entry,
  std::string const& column_name,
  std::vector<parquet_split_info::fixed_page_row_range> const& row_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // wdy start
  auto const stage_start = std::chrono::steady_clock::now();
  auto const log_stage   = [&](std::size_t rows, std::size_t pieces, std::string_view path) {
    auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - stage_start)
                               .count();
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] stage_timing stage=cache_column_materialize column={} rows={} "
      "ranges={} pieces={} path={} duration_us={}",
      column_name,
      rows,
      row_ranges.size(),
      pieces,
      path,
      duration_us);
  };
  // wdy end
  auto chunks_it = entry.data_batches_by_column.find(column_name);
  if (chunks_it == entry.data_batches_by_column.end()) {
    throw std::runtime_error("[fixed-page-cache] cached column '" + column_name +
                             "' missing from pinned entry");
  }
  auto const& chunks = chunks_it->second;

  if (!row_ranges.empty()) {
    bool contiguous              = true;
    auto const contiguous_start  = row_ranges.front().row_offset;
    auto expected_row_offset     = contiguous_start;
    std::size_t contiguous_rows  = 0;
    for (auto const& range : row_ranges) {
      if (range.row_offset != expected_row_offset) {
        contiguous = false;
        break;
      }
      expected_row_offset += range.num_rows;
      contiguous_rows += range.num_rows;
    }

    if (contiguous && contiguous_rows != 0) {
      auto const contiguous_end = contiguous_start + contiguous_rows;
      std::size_t chunk_base    = 0;
      for (auto const& chunk : chunks) {
        if (!chunk) { throw std::runtime_error("[fixed-page-cache] null cached column chunk"); }
        auto const chunk_rows = static_cast<std::size_t>(chunk->size());
        auto const chunk_end  = chunk_base + chunk_rows;
        if (contiguous_start >= chunk_base && contiguous_end <= chunk_end) {
          auto const local_start = contiguous_start - chunk_base;
          auto const local_end   = contiguous_end - chunk_base;
          // wdy start
          auto source_view = chunk->view();
          std::string_view copy_path = "full_chunk_copy";
          if (local_start != 0 || local_end != chunk_rows) {
            if (local_end > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
              throw std::overflow_error("[fixed-page-cache] cached slice exceeds cudf::size_type");
            }
            auto sliced = cudf::slice(
              source_view,
              {static_cast<cudf::size_type>(local_start), static_cast<cudf::size_type>(local_end)},
              stream);
            if (sliced.empty()) { continue; }
            source_view = sliced.front();
            copy_path = "contiguous_copy";
          }
          SIRIUS_LOG_DEBUG(
            "[fixed-page-cache] hybrid_reuse_fast_contiguous_slice column={} rows={}",
            column_name,
            contiguous_rows);
          auto out = std::make_unique<cudf::column>(source_view, stream, mr);
          log_stage(contiguous_rows, 1, copy_path);
          return out;
          // wdy end
        }
        chunk_base = chunk_end;
      }
    }
  }

  std::vector<cudf::column_view> piece_views;
  piece_views.reserve(row_ranges.size());

  for (auto const& range : row_ranges) {
    auto const range_end   = range.row_offset + range.num_rows;
    std::size_t chunk_base = 0;
    std::size_t remaining  = range.num_rows;

    for (auto const& chunk : chunks) {
      if (!chunk) { throw std::runtime_error("[fixed-page-cache] null cached column chunk"); }
      auto const chunk_rows = static_cast<std::size_t>(chunk->size());
      auto const chunk_end  = chunk_base + chunk_rows;
      if (range.row_offset >= chunk_end || range_end <= chunk_base) {
        chunk_base = chunk_end;
        continue;
      }

      auto const local_start = range.row_offset > chunk_base ? range.row_offset - chunk_base : 0;
      auto const local_end   = std::min(range_end, chunk_end) - chunk_base;
      // wdy start
      if (local_start == 0 && local_end == chunk_rows) {
        piece_views.push_back(chunk->view());
        remaining -= local_end - local_start;
      } else {
        if (local_end > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
          throw std::overflow_error("[fixed-page-cache] cached slice exceeds cudf::size_type");
        }
        auto sliced = cudf::slice(
          chunk->view(),
          {static_cast<cudf::size_type>(local_start), static_cast<cudf::size_type>(local_end)},
          stream);
        if (!sliced.empty()) {
          piece_views.push_back(sliced.front());
          remaining -= local_end - local_start;
        }
      }
      // wdy end
      if (remaining == 0) { break; }
      chunk_base = chunk_end;
    }

    if (remaining != 0) {
      throw std::runtime_error(
        "[fixed-page-cache] pinned entry does not cover requested rows for column '" + column_name +
        "'");
    }
  }

  if (piece_views.empty()) {
    throw std::runtime_error("[fixed-page-cache] no cached slices produced for column '" +
                             column_name + "'");
  }
  if (piece_views.size() == 1) {
    auto out = std::make_unique<cudf::column>(piece_views.front(), stream, mr);
    // wdy start
    log_stage(fixed_page_row_count(row_ranges), piece_views.size(), "single_piece_copy");
    // wdy end
    return out;
  }
  auto out = cudf::concatenate(
    cudf::host_span<cudf::column_view const>(piece_views.data(), piece_views.size()), stream, mr);
  // wdy start
  log_stage(fixed_page_row_count(row_ranges), piece_views.size(), "concatenate");
  // wdy end
  return out;
}

struct fixed_page_spliced_view {
  std::unique_ptr<cudf::table> parquet_table;
  std::vector<std::shared_ptr<cudf::column>> cached_column_owners;
  cudf::table_view view;
};

std::optional<cudf::column_view> contiguous_cached_fixed_column_view(
  scan_manager::pinned_entry const& entry,
  std::string const& column_name,
  std::vector<parquet_split_info::fixed_page_row_range> const& row_ranges,
  rmm::cuda_stream_view stream,
  std::vector<std::shared_ptr<cudf::column>>& owners)
{
  // wdy start
  auto const stage_start = std::chrono::steady_clock::now();
  // wdy end
  auto chunks_it = entry.data_batches_by_column.find(column_name);
  if (chunks_it == entry.data_batches_by_column.end() || row_ranges.empty()) { return std::nullopt; }

  bool contiguous             = true;
  auto const start            = row_ranges.front().row_offset;
  auto expected_row_offset    = start;
  std::size_t total_rows      = 0;
  for (auto const& range : row_ranges) {
    if (range.row_offset != expected_row_offset) {
      contiguous = false;
      break;
    }
    expected_row_offset += range.num_rows;
    total_rows += range.num_rows;
  }
  if (!contiguous || total_rows == 0) { return std::nullopt; }

  auto const end = start + total_rows;
  std::size_t chunk_base = 0;
  for (auto const& chunk : chunks_it->second) {
    if (!chunk) { throw std::runtime_error("[fixed-page-cache] null cached column chunk"); }
    auto const chunk_rows = static_cast<std::size_t>(chunk->size());
    auto const chunk_end  = chunk_base + chunk_rows;
    if (start >= chunk_base && end <= chunk_end) {
      auto const local_start = start - chunk_base;
      auto const local_end   = end - chunk_base;
      // wdy start
      auto source_view = chunk->view();
      std::string_view view_path = "full_chunk_view";
      if (local_start != 0 || local_end != chunk_rows) {
        if (local_end > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
          throw std::overflow_error("[fixed-page-cache] cached slice exceeds cudf::size_type");
        }
        auto sliced = cudf::slice(
          source_view,
          {static_cast<cudf::size_type>(local_start), static_cast<cudf::size_type>(local_end)},
          stream);
        if (sliced.empty()) { return std::nullopt; }
        source_view = sliced.front();
        view_path = "contiguous_view";
      }
      owners.push_back(chunk);
      auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - stage_start)
                                 .count();
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] stage_timing stage=cache_column_view column={} rows={} ranges={} "
        "pieces=1 path={} duration_us={}",
        column_name,
        total_rows,
        row_ranges.size(),
        view_path,
        duration_us);
      return source_view;
      // wdy end
    }
    chunk_base = chunk_end;
  }
  return std::nullopt;
}

std::shared_ptr<::cucascade::data_batch> make_fixed_page_spliced_view_batch(
  fixed_page_spliced_view&& spliced,
  ::cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  // wdy start
  auto owners = std::move(spliced.cached_column_owners);
  std::size_t alloc_size = 0;
  for (auto const& owner : owners) {
    if (owner) { alloc_size += owner->alloc_size(); }
  }

  if (spliced.parquet_table) {
    auto parquet_columns = spliced.parquet_table->release();
    owners.reserve(owners.size() + parquet_columns.size());
    for (auto& column : parquet_columns) {
      if (!column) { continue; }
      alloc_size += column->alloc_size();
      owners.emplace_back(std::shared_ptr<cudf::column>(std::move(column)));
    }
  }

  auto gpu_repr = std::make_unique<::cucascade::gpu_table_representation>(
    spliced.view, std::move(owners), alloc_size, mem_space, stream);
  return std::make_shared<::cucascade::data_batch>(::sirius::get_next_batch_id(),
                                                   std::move(gpu_repr));
  // wdy end
}

std::optional<fixed_page_spliced_view> try_splice_fixed_page_cached_columns_view(
  std::unique_ptr<cudf::table>& parquet_table,
  parquet_split_info const& split,
  scan_manager::pinned_entry const& entry,
  std::vector<bool> const& cached_by_data_index,
  rmm::cuda_stream_view stream)
{
  if (!split.fixed_page_reuse || !parquet_table) { return std::nullopt; }
  // wdy start
  auto const stage_start = std::chrono::steady_clock::now();
  // wdy end

  fixed_page_spliced_view result;
  auto parquet_view = parquet_table->view();
  std::vector<cudf::column_view> output_views(cached_by_data_index.size());
  std::size_t parquet_pos = 0;

  for (std::size_t data_idx = 0; data_idx < cached_by_data_index.size(); ++data_idx) {
    if (cached_by_data_index[data_idx]) {
      auto const& column_name = split.plan->data_columns.at(data_idx).name;
      auto cached_view = contiguous_cached_fixed_column_view(
        entry, column_name, split.fixed_page_reuse->row_ranges, stream, result.cached_column_owners);
      if (!cached_view) {
        // wdy start
        auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - stage_start)
                                   .count();
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] stage_timing stage=splice_view_build status=fallback "
          "cached_cols={} parquet_cols={} duration_us={}",
          split.fixed_page_reuse->cached_data_indices.size(),
          split.fixed_page_reuse->parquet_data_indices.size(),
          duration_us);
        // wdy end
        return std::nullopt;
      }
      output_views[data_idx] = *cached_view;
    } else {
      if (parquet_pos >= static_cast<std::size_t>(parquet_view.num_columns())) {
        throw std::runtime_error(
          "[fixed-page-cache] parquet reader returned fewer columns than expected");
      }
      output_views[data_idx] = parquet_view.column(static_cast<cudf::size_type>(parquet_pos++));
    }
  }
  if (parquet_pos != static_cast<std::size_t>(parquet_view.num_columns())) {
    throw std::runtime_error("[fixed-page-cache] parquet reader returned extra columns");
  }

  result.view          = cudf::table_view(output_views);
  result.parquet_table = std::move(parquet_table);
  // wdy start
  auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - stage_start)
                             .count();
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] stage_timing stage=splice_view_build status=success cached_cols={} "
    "parquet_cols={} rows={} columns={} duration_us={}",
    split.fixed_page_reuse->cached_data_indices.size(),
    split.fixed_page_reuse->parquet_data_indices.size(),
    result.view.num_rows(),
    result.view.num_columns(),
    duration_us);
  // wdy end
  return std::move(result);
}

std::unique_ptr<cudf::table> splice_fixed_page_cached_columns(
  std::unique_ptr<cudf::table> parquet_table,
  parquet_split_info const& split,
  scan_manager::pinned_entry const& entry,
  std::vector<bool> const& cached_by_data_index,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (!split.fixed_page_reuse) { return parquet_table; }
  auto const& reuse = *split.fixed_page_reuse;
  // wdy start
  auto const stage_start = std::chrono::steady_clock::now();
  // wdy end
  if (reuse.preferred_device_id >= 0 && reuse.preferred_device_id != mem_space.get_device_id()) {
    throw std::runtime_error("[fixed-page-cache] hybrid split scheduled on GPU " +
                             std::to_string(mem_space.get_device_id()) +
                             " but cached pages prefer GPU " +
                             std::to_string(reuse.preferred_device_id));
  }

  auto parquet_cols =
    parquet_table ? parquet_table->release() : std::vector<std::unique_ptr<cudf::column>>{};
  std::vector<std::unique_ptr<cudf::column>> output_cols(cached_by_data_index.size());
  std::size_t parquet_pos = 0;

  for (std::size_t data_idx = 0; data_idx < cached_by_data_index.size(); ++data_idx) {
    if (cached_by_data_index[data_idx]) {
      auto const& column_name = split.plan->data_columns.at(data_idx).name;
      output_cols[data_idx] =
        materialize_cached_fixed_column(entry, column_name, reuse.row_ranges, stream, mr);
    } else {
      if (parquet_pos >= parquet_cols.size()) {
        throw std::runtime_error(
          "[fixed-page-cache] parquet reader returned fewer columns than expected");
      }
      output_cols[data_idx] = std::move(parquet_cols[parquet_pos++]);
    }
  }
  if (parquet_pos != parquet_cols.size()) {
    throw std::runtime_error("[fixed-page-cache] parquet reader returned extra columns");
  }

  auto out = std::make_unique<cudf::table>(std::move(output_cols));
  // wdy start
  auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - stage_start)
                             .count();
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] stage_timing stage=splice_materialize target_gpu={} cached_cols={} "
    "parquet_cols={} rows={} columns={} duration_us={}",
    mem_space.get_device_id(),
    reuse.cached_data_indices.size(),
    reuse.parquet_data_indices.size(),
    out->num_rows(),
    out->num_columns(),
    duration_us);
  // wdy end
  return out;
}

std::unique_ptr<cudf::table> splice_filtered_fixed_page_cached_columns(
  std::unique_ptr<cudf::table> parquet_table,
  parquet_split_info const& split,
  scan_manager::pinned_entry const& entry,
  std::vector<bool> const& cached_by_data_index,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  sirius::ast::node const& filter_ast)
{
  if (!split.fixed_page_reuse) { return parquet_table; }
  auto const& reuse = *split.fixed_page_reuse;
  auto const stage_start = std::chrono::steady_clock::now();

  if (reuse.preferred_device_id >= 0 && reuse.preferred_device_id != mem_space.get_device_id()) {
    throw std::runtime_error("[fixed-page-cache] filtered hybrid split scheduled on GPU " +
                             std::to_string(mem_space.get_device_id()) +
                             " but cached pages prefer GPU " +
                             std::to_string(reuse.preferred_device_id));
  }

  auto const parquet_rows = parquet_table ? parquet_table->num_rows() : 0;
  auto parquet_cols =
    parquet_table ? parquet_table->release() : std::vector<std::unique_ptr<cudf::column>>{};
  auto const expected_parquet_cols = reuse.parquet_data_indices.size();
  if (parquet_cols.size() < expected_parquet_cols) {
    throw std::runtime_error(
      "[fixed-page-cache] filtered parquet reader returned fewer columns than expected");
  }

  std::vector<std::unique_ptr<cudf::column>> cached_cols;
  cached_cols.reserve(reuse.cached_data_indices.size());
  std::unordered_map<std::size_t, std::size_t> filter_ref_remap;
  for (std::size_t cached_pos = 0; cached_pos < reuse.cached_data_indices.size(); ++cached_pos) {
    auto const data_idx     = reuse.cached_data_indices[cached_pos];
    auto const& column_name = split.plan->data_columns.at(data_idx).name;
    cached_cols.push_back(
      materialize_cached_fixed_column(entry, column_name, reuse.row_ranges, stream, mr));
    filter_ref_remap.emplace(data_idx, cached_pos);
  }

  auto cached_table    = std::make_unique<cudf::table>(std::move(cached_cols));
  auto remapped_filter = remap_filter_references(filter_ast, filter_ref_remap);
  sirius::gpu_expression_executor exec(
    remapped_filter.get(), cudf::get_current_device_resource_ref(), stream);

  auto const filter_start      = std::chrono::steady_clock::now();
  auto const filter_input_rows = cached_table->num_rows();
  auto filtered_cached_table   = exec.select(cached_table->view());
  auto const filter_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - filter_start)
                                    .count();
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] stage_timing stage=cached_filter_select target_gpu={} input_rows={} "
    "output_rows={} input_columns={} output_columns={} duration_us={}",
    mem_space.get_device_id(),
    filter_input_rows,
    filtered_cached_table->num_rows(),
    cached_table->num_columns(),
    filtered_cached_table->num_columns(),
    filter_duration_us);

  if (filtered_cached_table->num_rows() != parquet_rows) {
    throw std::runtime_error("[fixed-page-cache] filtered cache/parquet row-count mismatch: " +
                             std::to_string(filtered_cached_table->num_rows()) + " vs " +
                             std::to_string(parquet_rows));
  }

  auto filtered_cached_cols = filtered_cached_table->release();
  std::vector<std::unique_ptr<cudf::column>> output_cols(cached_by_data_index.size());
  std::size_t cached_pos = 0;
  std::size_t parquet_pos = 0;

  for (std::size_t data_idx = 0; data_idx < cached_by_data_index.size(); ++data_idx) {
    if (cached_by_data_index[data_idx]) {
      if (cached_pos >= filtered_cached_cols.size()) {
        throw std::runtime_error(
          "[fixed-page-cache] filtered cached table returned fewer columns than expected");
      }
      output_cols[data_idx] = std::move(filtered_cached_cols[cached_pos++]);
    } else {
      if (parquet_pos >= expected_parquet_cols) {
        throw std::runtime_error(
          "[fixed-page-cache] filtered parquet output consumed more columns than expected");
      }
      output_cols[data_idx] = std::move(parquet_cols[parquet_pos++]);
    }
  }

  if (cached_pos != filtered_cached_cols.size()) {
    throw std::runtime_error("[fixed-page-cache] filtered cached table returned extra columns");
  }
  if (parquet_pos != expected_parquet_cols) {
    throw std::runtime_error("[fixed-page-cache] filtered parquet reader returned unused output columns");
  }

  auto out = std::make_unique<cudf::table>(std::move(output_cols));
  auto const duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - stage_start)
                             .count();
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] stage_timing stage=filtered_splice_materialize target_gpu={} "
    "cached_cols={} parquet_cols={} reader_extra_filter_cols={} rows={} columns={} duration_us={}",
    mem_space.get_device_id(),
    reuse.cached_data_indices.size(),
    reuse.parquet_data_indices.size(),
    reuse.reader_extra_filter_data_indices.size(),
    out->num_rows(),
    out->num_columns(),
    duration_us);
  return out;
}

// wdy start
std::string join_strings(std::vector<std::string> const& values, char sep)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) { out << sep; }
    out << values[i];
  }
  return out.str();
}

std::string join_row_group_indices(std::vector<cudf::size_type> const& values)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) { out << ","; }
    out << values[i];
  }
  return out.str();
}

std::string scan_audit_file_paths(std::vector<row_group_slice> const& slices)
{
  std::vector<std::string> files;
  files.reserve(slices.size());
  for (auto const& slice : slices) {
    files.push_back(slice.file_path);
  }
  return join_strings(files, '|');
}

std::string scan_audit_row_groups(std::vector<row_group_slice> const& slices)
{
  std::vector<std::string> groups;
  groups.reserve(slices.size());
  for (auto const& slice : slices) {
    groups.push_back(join_row_group_indices(slice.row_group_indices));
  }
  return join_strings(groups, '|');
}

std::size_t scan_audit_compressed_bytes(std::vector<row_group_slice> const& slices)
{
  std::size_t total = 0;
  for (auto const& slice : slices) {
    total += slice.reserved_compressed_bytes;
  }
  return total;
}

std::size_t scan_audit_uncompressed_bytes(std::vector<row_group_slice> const& slices)
{
  std::size_t total = 0;
  for (auto const& slice : slices) {
    total += slice.reserved_uncompressed_bytes;
  }
  return total;
}

std::string scan_audit_column_bytes(std::vector<row_group_slice> const& slices,
                                    std::vector<std::string> const& column_names)
{
  struct byte_pair {
    std::size_t compressed   = 0;
    std::size_t uncompressed = 0;
  };

  std::vector<byte_pair> totals(column_names.size());
  for (auto const& slice : slices) {
    auto const& metadata = *slice.file_metadata;
    std::vector<std::vector<std::size_t>> leaf_indices;
    leaf_indices.reserve(column_names.size());
    for (auto const& column_name : column_names) {
      leaf_indices.push_back(detail::leaf_indices_for_column(metadata, column_name));
    }

    for (auto const rg_idx : slice.row_group_indices) {
      auto const& row_group = metadata.row_groups[rg_idx];
      for (std::size_t col_idx = 0; col_idx < leaf_indices.size(); ++col_idx) {
        for (auto const leaf_idx : leaf_indices[col_idx]) {
          auto const& column_metadata = row_group.columns[leaf_idx].meta_data;
          totals[col_idx].compressed +=
            static_cast<std::size_t>(column_metadata.total_compressed_size);
          totals[col_idx].uncompressed +=
            static_cast<std::size_t>(column_metadata.total_uncompressed_size);
        }
      }
    }
  }

  std::ostringstream out;
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    if (i != 0) { out << ","; }
    out << column_names[i] << ":" << totals[i].compressed << ":" << totals[i].uncompressed;
  }
  return out.str();
}
// wdy end

}  // namespace

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<io::gpu_ingestible> parquet_ingestible_table_info::make_ingestible(
  std::unique_ptr<io::ingestible_table_info> self, scan_manager::sirius_scan_manager const& mgr)
{
  return std::make_shared<parquet_gpu_ingestible>(std::move(self), mgr);
}

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
parquet_gpu_ingestible::parquet_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info,
                                               scan_manager::sirius_scan_manager const& mgr)
  : io::gpu_ingestible(std::move(info)), _scan_manager(&mgr)
{
  auto const& bind = static_cast<parquet_ingestible_table_info const&>(table_info());

  // Any non-trivial scan shape — reader-side projection, filter pushdown, or hive-partition
  // injection — needs column names. Matches parquet_split_provider's ctor invariant.
  bool const needs_names = !bind.projection_ids.empty() ||
                           (bind.table_filters && !bind.table_filters->filters.empty()) ||
                           !bind.partition_indices.empty();
  if (needs_names && bind.names.empty()) {
    throw sirius::internal_exception(
      "[parquet_gpu_ingestible] Projection, filter pushdown, or hive partitions "
      "require column names to be provided.");
  }

  _plan = std::make_shared<scan_plan const>(build_scan_plan(bind.column_ids,
                                                            bind.projection_ids,
                                                            bind.names,
                                                            bind.returned_types,
                                                            bind.scan_output_arity,
                                                            bind.partition_indices));

  // AST translation deferred to materialize_table so a task-local stream is used.
  // Filters on hive-partition columns are dropped — those columns aren't in the
  // parquet file (DuckDB prunes them at the file-list level already).
  if (bind.table_filters && !bind.table_filters->filters.empty()) {
    auto duckdb_expression =
      sirius::op::convert_table_filters_to_expression(*bind.table_filters,
                                                      bind.column_ids,
                                                      bind.returned_types,
                                                      _plan->batch_position_by_column_id,
                                                      _plan->partition_primary_indices);
    if (duckdb_expression) { _duckdb_filter_expression = std::move(duckdb_expression); }
  }

  _file_paths             = bind.resolved_file_paths;
  _approximate_batch_size = bind.approximate_batch_size;
  _max_file_processed     = bind.max_file_processed;
  _total_files            = _file_paths.size();

  initialize_fixed_page_cache();

  for (std::size_t start = 0; start < _total_files; start += _max_file_processed) {
    auto const end = std::min(start + _max_file_processed, _total_files);
    file_batch batch;
    batch.file_paths.assign(_file_paths.begin() + static_cast<std::ptrdiff_t>(start),
                            _file_paths.begin() + static_cast<std::ptrdiff_t>(end));
    _batches.push_back(std::move(batch));
  }
}

parquet_gpu_ingestible::~parquet_gpu_ingestible() = default;

void parquet_gpu_ingestible::initialize_fixed_page_cache()
{
  if (!fixed_page_reuse_enabled()) { return; }
  if (_scan_manager == nullptr || !_plan) { return; }
  if (_file_paths.size() != 1) {
    SIRIUS_LOG_DEBUG(
      "[fixed-page-cache] hybrid reuse skipped: prototype currently requires a single file");
    return;
  }
  if (_plan->data_columns.empty()) { return; }

  for (auto const& [pinned_name, entry] : _scan_manager->get_pinned_entries()) {
    if (entry.tier != cucascade::memory::Tier::GPU) { continue; }
    if (entry.is_partial) {
      SIRIUS_LOG_DEBUG(
        "[fixed-page-cache] hybrid reuse skipped for pinned='{}': partial pin entries are not "
        "used by the correctness-preserving prototype",
        pinned_name);
      continue;
    }
    if (!same_file_paths_exact(entry.file_paths, _file_paths)) { continue; }
    if (entry.chunk_memory_spaces.empty()) { continue; }

    fixed_page_cache_state state;
    state.pinned_name = pinned_name;
    state.entry       = &entry;
    state.cached_by_data_index.assign(_plan->data_columns.size(), false);

    for (std::size_t data_idx = 0; data_idx < _plan->data_columns.size(); ++data_idx) {
      auto const& col = _plan->data_columns[data_idx];
      auto page_it    = entry.fixed_width_pages_by_column.find(col.name);
      auto chunk_it   = entry.data_batches_by_column.find(col.name);
      if (page_it == entry.fixed_width_pages_by_column.end() ||
          chunk_it == entry.data_batches_by_column.end() || page_it->second.empty() ||
          chunk_it->second.empty()) {
        state.parquet_data_indices.push_back(data_idx);
        state.parquet_column_names.push_back(col.name);
        continue;
      }

      state.cached_by_data_index[data_idx] = true;
      state.cached_data_indices.push_back(data_idx);
      state.cached_row_width_bytes += page_it->second.front().element_size_bytes;
    }

    if (state.cached_data_indices.empty()) { continue; }
    if (state.parquet_data_indices.empty()) {
      SIRIUS_LOG_DEBUG(
        "[fixed-page-cache] hybrid reuse skipped for pinned='{}': all requested columns are "
        "cached; existing pinned-table path should handle this case",
        pinned_name);
      continue;
    }

    SIRIUS_LOG_INFO(
      "[fixed-page-cache] hybrid candidate pinned='{}' cached_cols={} parquet_cols={} pages={} "
      "page_bytes={} cached_devices={}",
      state.pinned_name,
      state.cached_data_indices.size(),
      state.parquet_data_indices.size(),
      fixed_page_count_for_columns(entry, state.cached_data_indices, *_plan),
      entry.fixed_width_page_size_bytes,
      fixed_page_candidate_device_count(entry));
    _fixed_page_cache = std::move(state);
    return;
  }
}

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool parquet_gpu_ingestible::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
parquet_gpu_ingestible::next_split_provider()
{
  auto const batch_idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (batch_idx >= _batches.size()) { return nullptr; }
  return [this, batch_idx]() {
    std::vector<std::unique_ptr<op::operator_data>> out;
    run_batch(_batches[batch_idx], out);
    return out;
  };
}

//===----------------------------------------------------------------------===//
// run_batch — ports parquet_split_provider::run_batch
//===----------------------------------------------------------------------===//
void parquet_gpu_ingestible::run_batch(file_batch const& batch,
                                       std::vector<std::unique_ptr<op::operator_data>>& out)
{
  auto stream = cudf::get_default_stream();

  auto const data_column_names = _plan->data_column_names();
  std::vector<std::size_t> reader_data_indices;
  std::vector<std::string> reader_column_names;
  if (_fixed_page_cache) {
    reader_data_indices = _fixed_page_cache->parquet_data_indices;
    reader_column_names = _fixed_page_cache->parquet_column_names;
  } else {
    reader_data_indices.reserve(data_column_names.size());
    reader_column_names.reserve(data_column_names.size());
    for (std::size_t i = 0; i < data_column_names.size(); ++i) {
      reader_data_indices.push_back(i);
      reader_column_names.push_back(data_column_names[i]);
    }
  }
  // wdy start
  auto full_reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  if (_plan->is_projected()) { full_reader_options->set_column_names(data_column_names); }
  auto const min_filtered_split_cached_bytes = fixed_page_min_filtered_split_cached_bytes();
  // wdy end
  auto reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());

  if (_fixed_page_cache) {
    reader_options->set_column_names(reader_column_names);
  } else if (_plan->is_projected()) {
    reader_options->set_column_names(data_column_names);
  }

  if (_scan_manager == nullptr) {
    throw std::runtime_error("parquet_gpu_ingestible: no scan_manager is wired.");
  }

  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  bool skip_pushdown_due_to_flba                                                 = false;
  std::unique_ptr<sirius::ast::node> sirius_filter_ast;
  fixed_page_pruning_constraints fixed_page_pruning_constraints_by_data_index;
  std::shared_ptr<cudf::io::parquet_reader_options> pruning_reader_options;
  if (_duckdb_filter_expression) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    if (_fixed_page_cache && fixed_page_pruning_enabled() && sirius_filter_ast) {
      extract_fixed_page_pruning_constraints(
        *sirius_filter_ast, fixed_page_pruning_constraints_by_data_index, *_plan);
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_pruning_constraints pinned='{}' constraints={} cached_cols={}",
        _fixed_page_cache->pinned_name,
        fixed_page_pruning_constraints_by_data_index.size(),
        _fixed_page_cache->cached_data_indices.size());
    }
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) {
      // FLBA-decimal pushdown probe — see parquet_split_provider.cpp:276-326.
      if (!batch.file_paths.empty()) {
        auto const& probe_path = batch.file_paths.front();
        try {
          // Resolve the probe file to a datasource (carries its own io backend,
          // cache and metadata) instead of reaching into io_context. Local
          // paths left unclaimed (use_sirius_datasource=false) probe through a
          // plain cudf datasource so filter pushdown is not lost.
          auto probe_ds = _scan_manager->create_datasource(probe_path);
          std::unique_ptr<cudf::io::datasource> probe_fallback;
          cudf::io::datasource* probe_src = probe_ds.get();
          if (probe_src == nullptr) {
            auto const probe_local = strip_file_uri(probe_path);
            if (has_uri_scheme(probe_local)) {
              throw std::runtime_error("no backend supports path: " + probe_path);
            }
            probe_fallback = cudf::io::datasource::create(probe_local);
            probe_src      = probe_fallback.get();
          }
          std::shared_ptr<cudf::io::parquet::FileMetaData const> probe_meta;
          if (probe_ds) {
            if (auto cached = probe_ds->metadata()) {
              if (auto pm = std::dynamic_pointer_cast<scan_manager::parquet_metadata>(cached)) {
                probe_meta = pm->file_metadata();
              }
            }
          }
          if (!probe_meta) {
            auto footer = cudf::io::parquet::fetch_footer_to_host(*probe_src);
            hybrid_scan_reader probe_reader(
              cudf::host_span<uint8_t const>(footer->data(), footer->size()), *reader_options);
            probe_meta = std::make_shared<cudf::io::parquet::FileMetaData const>(
              probe_reader.parquet_metadata());
          }
          for (auto const& elem : probe_meta->schema) {
            bool const is_decimal =
              (elem.converted_type.has_value() &&
               *elem.converted_type == cudf::io::parquet::ConvertedType::DECIMAL) ||
              (elem.logical_type.has_value() &&
               elem.logical_type->type == cudf::io::parquet::LogicalType::DECIMAL);
            if (!is_decimal) { continue; }
            if (elem.type == cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY ||
                elem.type == cudf::io::parquet::Type::BYTE_ARRAY) {
              skip_pushdown_due_to_flba = true;
              break;
            }
          }
        } catch (std::exception const& e) {
          SIRIUS_LOG_DEBUG(
            "[parquet_gpu_ingestible] FLBA-decimal probe failed ({}); proceeding without "
            "pushdown",
            e.what());
          skip_pushdown_due_to_flba = true;
        }
      }

      if (!skip_pushdown_due_to_flba) {
        if (_fixed_page_cache) {
          pruning_reader_options =
            std::make_shared<cudf::io::parquet_reader_options>(*full_reader_options);
          pruning_reader_options->set_filter(ast_expression->back());
          SIRIUS_LOG_DEBUG(
            "[fixed-page-cache] Using translated filter with full scan projection for "
            "row-group pruning only; row filtering runs after cached columns are spliced.");
        } else {
          reader_options->set_filter(ast_expression->back());
          SIRIUS_LOG_DEBUG(
            "[parquet_gpu_ingestible] Translated filter expression for row group pruning.");
        }
      } else {
        SIRIUS_LOG_DEBUG(
          "[parquet_gpu_ingestible] Skipping row-group pruning pushdown: FLBA-decimal file.");
      }
    } else {
      SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] AST translation failed for row group pruning.");
    }
  }

  std::vector<std::size_t> fixed_page_filter_reference_data_indices;
  bool fixed_page_filtered_reuse_possible = false;
  if (_fixed_page_cache && fixed_page_filtered_reuse_enabled() && sirius_filter_ast &&
      ast_expression && !skip_pushdown_due_to_flba) {
    fixed_page_filter_reference_data_indices = filter_reference_data_indices(*sirius_filter_ast);
    fixed_page_filtered_reuse_possible = filter_references_are_cached(
      fixed_page_filter_reference_data_indices, _fixed_page_cache->cached_by_data_index);

    if (fixed_page_filtered_reuse_possible) {
      std::unordered_set<std::size_t> reader_data_index_set(reader_data_indices.begin(),
                                                            reader_data_indices.end());
      for (auto const data_idx : fixed_page_filter_reference_data_indices) {
        if (reader_data_index_set.insert(data_idx).second) {
          reader_data_indices.push_back(data_idx);
          reader_column_names.push_back(_plan->data_columns.at(data_idx).name);
        }
      }
      reader_options->set_column_names(reader_column_names);
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] filtered_reuse_enabled pinned='{}' filter_refs={} "
        "reader_output_cols={} reader_extra_filter_cols={}",
        _fixed_page_cache->pinned_name,
        fixed_page_filter_reference_data_indices.size(),
        _fixed_page_cache->parquet_data_indices.size(),
        reader_column_names.size() - _fixed_page_cache->parquet_data_indices.size());
    } else {
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] filtered_reuse_disabled pinned='{}' reason=filter_refs_not_all_cached "
        "filter_refs={} cached_cols={}",
        _fixed_page_cache->pinned_name,
        fixed_page_filter_reference_data_indices.size(),
        _fixed_page_cache->cached_data_indices.size());
    }
  }

  rg_accumulator accum;
  std::optional<int> accum_fixed_page_device_id;
  // wdy start
  std::optional<std::size_t> accum_fixed_page_chunk_index;
  std::optional<std::size_t> accum_fixed_page_next_row_offset;
  bool const align_fixed_page_view_splits =
    _fixed_page_cache && fixed_page_view_aligned_splits_enabled();
  // wdy end
  bool const needs_post_processing = needs_output_assembly(*_plan);

  auto build_post_filter_info =
    [&accum, needs_post_processing]() -> std::unique_ptr<io::post_filter_and_projection_info> {
    if (!needs_post_processing) { return nullptr; }
    auto info              = std::make_unique<parquet_post_filter_and_projection_info>();
    info->partition_values = accum.partition_values.value_or(std::vector<std::string>{});
    return info;
  };

  auto flush = [&](std::shared_ptr<cudf::io::parquet_reader_options> shared_opts,
                   std::shared_ptr<scan_plan const> shared_plan) {
    if (accum.slices.empty()) { return; }
    auto split_info              = std::make_unique<parquet_split_info>();
    split_info->rg_slices        = std::move(accum.slices);
    split_info->plan             = std::move(shared_plan);
    split_info->needs_assembly   = needs_post_processing;
    split_info->partition_values = accum.partition_values.value_or(std::vector<std::string>{});

    std::optional<int> preferred_device_id;
    bool skip_fixed_page_reuse_for_filter = false;
    if (_fixed_page_cache && !accum.fixed_page_row_ranges.empty()) {
      auto device_choice = choose_fixed_page_device(*_fixed_page_cache->entry,
                                                    _fixed_page_cache->cached_data_indices,
                                                    *_plan,
                                                    accum.fixed_page_row_ranges);
      if (device_choice.device_id < 0 || device_choice.cached_bytes == 0) {
        throw std::runtime_error(
          "[fixed-page-cache] hybrid split has row ranges but no overlapping cached pages");
      }
      if (device_choice.device_count > 1) {
        throw std::runtime_error(
          "[fixed-page-cache] hybrid split spans multiple cached GPUs; split construction should "
          "flush at cached-device boundaries");
      }

      if (_duckdb_filter_expression && min_filtered_split_cached_bytes != 0 &&
          device_choice.cached_bytes < min_filtered_split_cached_bytes) {
        skip_fixed_page_reuse_for_filter = true;
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] hybrid_reuse_skip pinned='{}' reason=filtered_split_cached_bytes "
          "cached_bytes={} threshold={} cached_cols={} parquet_cols={}",
          _fixed_page_cache->pinned_name,
          device_choice.cached_bytes,
          min_filtered_split_cached_bytes,
          _fixed_page_cache->cached_data_indices.size(),
          _fixed_page_cache->parquet_data_indices.size());
      }

      if (!skip_fixed_page_reuse_for_filter && !fixed_page_filtered_reuse_possible &&
          _duckdb_filter_expression && ast_expression && !skip_pushdown_due_to_flba &&
          fixed_page_pruning_enabled() && !fixed_page_pruning_constraints_by_data_index.empty()) {
        auto const pruning_summary = summarize_fixed_page_pruning(
          *_fixed_page_cache->entry,
          _fixed_page_cache->cached_data_indices,
          *_plan,
          accum.fixed_page_row_ranges,
          fixed_page_pruning_constraints_by_data_index);
        bool const skip_partial = fixed_page_pruning_skip_partial_enabled();
        bool const should_preserve_pushdown =
          pruning_summary.pages_considered != 0 &&
          (pruning_summary.all_fail_pages != 0 ||
           (skip_partial && pruning_summary.partial_pages != 0));
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] page_pruning_decision pinned='{}' matched_cols={} pages={} "
          "all_fail={} all_pass={} partial={} unknown={} skip_partial={} action={}",
          _fixed_page_cache->pinned_name,
          pruning_summary.matched_columns,
          pruning_summary.pages_considered,
          pruning_summary.all_fail_pages,
          pruning_summary.all_pass_pages,
          pruning_summary.partial_pages,
          pruning_summary.unknown_pages,
          skip_partial,
          should_preserve_pushdown ? "skip_reuse_preserve_pushdown" : "keep_reuse");
        if (should_preserve_pushdown) { skip_fixed_page_reuse_for_filter = true; }
      }

      if (!skip_fixed_page_reuse_for_filter) {
        auto reuse                  = std::make_unique<parquet_split_info::fixed_page_reuse_info>();
        reuse->pinned_name                       = _fixed_page_cache->pinned_name;
        reuse->row_ranges                        = std::move(accum.fixed_page_row_ranges);
        reuse->cached_data_indices               = _fixed_page_cache->cached_data_indices;
        reuse->parquet_data_indices              = _fixed_page_cache->parquet_data_indices;
        reuse->reader_extra_filter_data_indices  = fixed_page_filtered_reuse_possible
                                                     ? fixed_page_filter_reference_data_indices
                                                     : std::vector<std::size_t>{};
        reuse->preferred_device_id               = device_choice.device_id;
        reuse->filtered_reuse                    = fixed_page_filtered_reuse_possible;
        split_info->fixed_page_cached_bytes = device_choice.cached_bytes;
        preferred_device_id                 = reuse->preferred_device_id;
        split_info->fixed_page_reuse        = std::move(reuse);
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] hybrid_reuse_split pinned='{}' cached_cols={} parquet_cols={} "
          "reader_extra_filter_cols={} filtered_reuse={} row_ranges={} cached_bytes={} "
          "best_device_bytes={} cached_devices={} preferred_gpu={}",
          _fixed_page_cache->pinned_name,
          _fixed_page_cache->cached_data_indices.size(),
          _fixed_page_cache->parquet_data_indices.size(),
          split_info->fixed_page_reuse->reader_extra_filter_data_indices.size(),
          split_info->fixed_page_reuse->filtered_reuse ? 1 : 0,
          split_info->fixed_page_reuse->row_ranges.size(),
          split_info->fixed_page_cached_bytes,
          device_choice.best_device_bytes,
          device_choice.device_count,
          device_choice.device_id);
      }
    }

    split_info->reader_options =
      skip_fixed_page_reuse_for_filter ? full_reader_options : std::move(shared_opts);
    bool const fixed_page_reuse_needs_post_filter =
      split_info->fixed_page_reuse != nullptr && !split_info->fixed_page_reuse->filtered_reuse;
    split_info->disable_filter_pushdown =
      skip_pushdown_due_to_flba || fixed_page_reuse_needs_post_filter;
    auto metadata = std::make_unique<io::scan_and_filter_metadata>(std::move(split_info),
                                                                   build_post_filter_info());
    auto input    = std::make_unique<scan_operator_input>(std::move(metadata));
    if (preferred_device_id) { input->set_preferred_device_id(*preferred_device_id); }
    out.push_back(std::move(input));
    accum.slices.clear();
    accum.fixed_page_row_ranges.clear();
    accum.total_uncompressed_bytes = 0;
    accum_fixed_page_device_id.reset();
    // wdy start
    accum_fixed_page_chunk_index.reset();
    accum_fixed_page_next_row_offset.reset();
    // wdy end
  };

  for (auto const& file_path : batch.file_paths) {
    if (!_plan->partition_columns.empty()) {
      std::vector<std::string> file_partition_values;
      file_partition_values.reserve(_plan->partition_columns.size());
      auto parsed = duckdb::HivePartitioning::Parse(file_path);
      for (auto const& pc : _plan->partition_columns) {
        auto it = parsed.find(pc.name);
        file_partition_values.push_back(it != parsed.end() ? it->second : std::string{});
      }
      if (accum.partition_values && *accum.partition_values != file_partition_values) {
        flush(reader_options, _plan);
      }
      accum.partition_values = std::move(file_partition_values);
    }

    // Resolve the file to a sirius_datasource — it carries its own io backend,
    // prefetch cache and cached metadata, so the ingestible no longer reaches
    // into io_context. Fall back to a plain cudf datasource only for local
    // paths no sirius backend claims.
    auto sirius_ds = _scan_manager->create_datasource(file_path);
    // file:// counts as local: create_datasource strips it before deciding the
    // local fallback, so the scheme check here must strip it too.
    auto const local_file_path = strip_file_uri(file_path);
    if (!sirius_ds && has_uri_scheme(local_file_path)) {
      throw std::runtime_error("[parquet_gpu_ingestible] no backend supports path: " + file_path);
    }

    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    std::shared_ptr<scan_manager::parquet_metadata> cached_parquet_metadata;
    std::size_t footer_byte_len = 0;
    std::unique_ptr<hybrid_scan_reader> reader_ptr;

    if (sirius_ds) {
      if (auto cached = sirius_ds->metadata()) {
        cached_parquet_metadata =
          std::dynamic_pointer_cast<scan_manager::parquet_metadata>(std::move(cached));
      }
    }

    if (cached_parquet_metadata) {
      file_metadata   = cached_parquet_metadata->file_metadata();
      footer_byte_len = cached_parquet_metadata->footer_byte_len();
      reader_ptr      = std::make_unique<hybrid_scan_reader>(*file_metadata, *reader_options);
    } else {
      // Local paths left unclaimed (use_sirius_datasource=false) read the
      // footer through a plain cudf datasource; the slice itself stays
      // datasource-less so materialize falls back to cudf/KvikIO.
      std::unique_ptr<cudf::io::datasource> footer_fallback;
      cudf::io::datasource* footer_src = sirius_ds.get();
      if (footer_src == nullptr) {
        footer_fallback = cudf::io::datasource::create(local_file_path);
        footer_src      = footer_fallback.get();
      }
      auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*footer_src);
      footer_byte_len    = footer_buffer->size();
      reader_ptr         = std::make_unique<hybrid_scan_reader>(
        cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
        *reader_options);
      file_metadata =
        std::make_shared<cudf::io::parquet::FileMetaData const>(reader_ptr->parquet_metadata());
    }
    auto& reader         = *reader_ptr;
    auto const& metadata = *file_metadata;

    std::vector<std::size_t> selected_chunk_indices;
    std::unordered_set<std::size_t> pure_filter_chunk_indices;
    if (_plan->is_projected() || _fixed_page_cache) {
      auto const pure_filter_positions = _plan->pure_filter_batch_positions();
      selected_chunk_indices.reserve(reader_column_names.size());
      for (std::size_t k = 0; k < reader_column_names.size(); ++k) {
        auto leaves = detail::leaf_indices_for_column(metadata, reader_column_names[k]);
        if (leaves.empty()) {
          throw std::runtime_error("[parquet_gpu_ingestible] Projected column '" +
                                   reader_column_names[k] +
                                   "' not found in parquet file: " + file_path);
        }
        auto const data_idx = reader_data_indices.at(k);
        bool const is_reader_extra_filter =
          _fixed_page_cache && k >= _fixed_page_cache->parquet_data_indices.size();
        bool const is_pure_filter = is_reader_extra_filter || pure_filter_positions.count(data_idx);
        for (auto const leaf : leaves) {
          selected_chunk_indices.push_back(leaf);
          if (is_pure_filter) { pure_filter_chunk_indices.insert(leaf); }
        }
      }
    }

    auto row_group_indices = reader.all_row_groups(*reader_options);
    if (ast_expression && !skip_pushdown_due_to_flba) {
      auto const rgs_before = row_group_indices.size();
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible] Row group pruning: file: {}\n"
        "                                                  before: {}",
        file_path,
        rgs_before);
      auto const& pruning_options =
        pruning_reader_options ? *pruning_reader_options : *reader_options;
      row_group_indices =
        reader.filter_row_groups_with_stats(row_group_indices, pruning_options, stream);
      SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible]                     after: {} (pruned {})",
                       row_group_indices.size(),
                       rgs_before - row_group_indices.size());
    }

    // Scan-side chunk prewarm: projection and row-group pruning are final for
    // this file, so hand the prefetch cache the merged column-chunk byte
    // ranges (plus parquet magic + footer) to stage while slices are built.
    // describe_parquet()'s insert stays metadata-only — this is the only
    // place ranges enter the cache, and only when the knob is on.
    if (sirius_ds && _scan_manager->chunk_prewarm_enabled() && !row_group_indices.empty()) {
      if (auto* cache = sirius_ds->io_ctx() ? sirius_ds->io_ctx()->cache() : nullptr) {
        using range_t = cudf::io::text::byte_range_info;

        auto chunk_ranges =
          reader.all_column_chunks_byte_ranges(row_group_indices, *reader_options);
        std::sort(chunk_ranges.begin(), chunk_ranges.end(), [](auto const& a, auto const& b) {
          return a.offset() < b.offset();
        });
        std::vector<range_t> merged;
        merged.reserve(chunk_ranges.size());
        if (!chunk_ranges.empty()) {
          auto cur_start = chunk_ranges[0].offset();
          auto cur_end   = cur_start + chunk_ranges[0].size();
          for (auto const& r : chunk_ranges) {
            auto const rs = r.offset();
            auto const re = rs + r.size();
            if (rs <= cur_end) {
              cur_end = std::max(cur_end, re);
            } else {
              merged.emplace_back(cur_start, cur_end - cur_start);
              cur_start = rs;
              cur_end   = re;
            }
          }
          merged.emplace_back(cur_start, cur_end - cur_start);
        }

        constexpr std::size_t FOOTER_TAIL_SIZE = 8;
        auto const file_size                   = sirius_ds->io_object()->size();
        auto const footer_off =
          static_cast<int64_t>(file_size - FOOTER_TAIL_SIZE - footer_byte_len);
        auto const footer_size = static_cast<int64_t>(FOOTER_TAIL_SIZE + footer_byte_len);

        std::vector<range_t> ranges;
        ranges.reserve(merged.size() + 2);
        ranges.emplace_back(0, 4);
        ranges.insert(ranges.end(), merged.begin(), merged.end());
        ranges.emplace_back(footer_off, footer_size);
        std::sort(ranges.begin(), ranges.end(), [](auto const& a, auto const& b) {
          return a.offset() < b.offset();
        });

        std::shared_ptr<sirius::io::sirius_io_object_metadata> metadata_to_store =
          cached_parquet_metadata
            ? nullptr
            : std::static_pointer_cast<sirius::io::sirius_io_object_metadata>(
                std::make_shared<scan_manager::parquet_metadata>(file_metadata, footer_byte_len));
        cache->insert(*sirius_ds->io_object(), std::move(metadata_to_store), ranges);
      }
    }

    auto const rg_start_offsets =
      _fixed_page_cache ? row_group_start_offsets(metadata) : std::vector<std::size_t>{};
    std::vector<cudf::size_type> cur_rgs;
    std::vector<parquet_split_info::fixed_page_row_range> cur_fixed_page_row_ranges;
    std::size_t cur_uncompressed_bytes = 0;
    std::size_t cur_compressed_bytes   = 0;

    auto seal_current_file = [&]() {
      if (cur_rgs.empty()) { return; }
      accum.slices.emplace_back(file_metadata,
                                file_path,
                                std::move(cur_rgs),
                                cur_uncompressed_bytes,
                                cur_compressed_bytes,
                                sirius_ds);
      accum.fixed_page_row_ranges.insert(accum.fixed_page_row_ranges.end(),
                                         cur_fixed_page_row_ranges.begin(),
                                         cur_fixed_page_row_ranges.end());
      accum.total_uncompressed_bytes += cur_uncompressed_bytes;
      cur_rgs.clear();
      cur_fixed_page_row_ranges.clear();
      cur_uncompressed_bytes = 0;
      cur_compressed_bytes   = 0;
    };

    auto rg_contribution = [&](cudf::io::parquet::RowGroup const& row_group) {
      std::size_t rg_uncompressed = 0;
      std::size_t rg_compressed   = 0;
      auto add_chunk = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
        auto const& column_metadata = chunk.meta_data;
        if (!is_pure_filter) {
          rg_uncompressed += static_cast<std::size_t>(column_metadata.total_uncompressed_size);
        }
        rg_compressed += static_cast<std::size_t>(column_metadata.total_compressed_size);
      };
      if (_plan->is_projected() || _fixed_page_cache) {
        for (auto const chunk_idx : selected_chunk_indices) {
          add_chunk(row_group.columns[chunk_idx], pure_filter_chunk_indices.contains(chunk_idx));
        }
      } else {
        for (auto const& chunk : row_group.columns) {
          add_chunk(chunk, false);
        }
      }
      return std::pair{rg_uncompressed, rg_compressed};
    };

    for (auto const rg_idx : row_group_indices) {
      auto const& row_group        = metadata.row_groups[rg_idx];
      auto const [rg_unc, rg_comp] = rg_contribution(row_group);

      std::optional<parquet_split_info::fixed_page_row_range> rg_fixed_page_row_range;
      std::optional<int> rg_fixed_page_device_id;
      // wdy start
      std::optional<std::size_t> rg_fixed_page_chunk_index;
      // wdy end
      if (_fixed_page_cache) {
        auto const rg_pos       = static_cast<std::size_t>(rg_idx);
        rg_fixed_page_row_range = parquet_split_info::fixed_page_row_range{
          rg_start_offsets.at(rg_pos), static_cast<std::size_t>(row_group.num_rows)};
        // wdy start
        rg_fixed_page_chunk_index =
          cached_chunk_index_for_range(*_fixed_page_cache->entry,
                                       _fixed_page_cache->cached_data_indices,
                                       *_plan,
                                       *rg_fixed_page_row_range);
        // wdy end
        std::vector<parquet_split_info::fixed_page_row_range> single_range{
          *rg_fixed_page_row_range};
        auto device_choice = choose_fixed_page_device(
          *_fixed_page_cache->entry, _fixed_page_cache->cached_data_indices, *_plan, single_range);
        if (device_choice.device_id >= 0 && device_choice.cached_bytes != 0 &&
            device_choice.device_count == 1) {
          rg_fixed_page_device_id = device_choice.device_id;
        } else {
          if (!accum.slices.empty() || !cur_rgs.empty()) {
            seal_current_file();
            flush(reader_options, _plan);
          }
          SIRIUS_LOG_INFO(
            "[fixed-page-cache] hybrid_reuse_row_group_fallback pinned='{}' row_group={} "
            "cached_devices={} cached_bytes={}",
            _fixed_page_cache->pinned_name,
            rg_idx,
            device_choice.device_count,
            device_choice.cached_bytes);
          cur_uncompressed_bytes += rg_unc;
          cur_compressed_bytes += rg_comp;
          cur_rgs.push_back(rg_idx);
          seal_current_file();
          flush(reader_options, _plan);
          continue;
        }
      }

      if (!accum.slices.empty() || !cur_rgs.empty()) {
        if (accum.total_uncompressed_bytes + cur_uncompressed_bytes + rg_unc >
            _approximate_batch_size) {
          seal_current_file();
          flush(reader_options, _plan);
        }
      }

      if (rg_fixed_page_device_id && accum_fixed_page_device_id &&
          *accum_fixed_page_device_id != *rg_fixed_page_device_id) {
        seal_current_file();
        flush(reader_options, _plan);
      }

      // wdy start
      if (align_fixed_page_view_splits && rg_fixed_page_row_range &&
          (!accum.slices.empty() || !cur_rgs.empty())) {
        bool should_flush_for_view = false;
        std::string_view reason    = "unknown";
        if (accum_fixed_page_next_row_offset &&
            rg_fixed_page_row_range->row_offset != *accum_fixed_page_next_row_offset) {
          should_flush_for_view = true;
          reason                = "row_gap";
        } else if (accum_fixed_page_chunk_index && rg_fixed_page_chunk_index &&
                   *accum_fixed_page_chunk_index != *rg_fixed_page_chunk_index) {
          should_flush_for_view = true;
          reason                = "chunk_boundary";
        } else if (accum_fixed_page_chunk_index.has_value() !=
                   rg_fixed_page_chunk_index.has_value()) {
          should_flush_for_view = true;
          reason                = "chunk_unknown_boundary";
        }

        if (should_flush_for_view) {
          SIRIUS_LOG_INFO(
            "[fixed-page-cache] view_aligned_split_flush pinned='{}' row_group={} reason={} "
            "prev_next_row={} prev_chunk={} next_row={} next_chunk={}",
            _fixed_page_cache->pinned_name,
            rg_idx,
            reason,
            accum_fixed_page_next_row_offset.value_or(0),
            accum_fixed_page_chunk_index.value_or(static_cast<std::size_t>(-1)),
            rg_fixed_page_row_range->row_offset,
            rg_fixed_page_chunk_index.value_or(static_cast<std::size_t>(-1)));
          seal_current_file();
          flush(reader_options, _plan);
        }
      }
      // wdy end

      cur_uncompressed_bytes += rg_unc;
      cur_compressed_bytes += rg_comp;
      cur_rgs.push_back(rg_idx);
      if (rg_fixed_page_row_range) {
        cur_fixed_page_row_ranges.push_back(*rg_fixed_page_row_range);
        accum_fixed_page_device_id = *rg_fixed_page_device_id;
        // wdy start
        accum_fixed_page_chunk_index = rg_fixed_page_chunk_index;
        accum_fixed_page_next_row_offset =
          rg_fixed_page_row_range->row_offset + rg_fixed_page_row_range->num_rows;
        // wdy end
      }
    }
    seal_current_file();
  }
  flush(reader_options, _plan);
}

//===----------------------------------------------------------------------===//
// materialize_table — ports read_table_from_metadata
//===----------------------------------------------------------------------===//
io::filtered_table parquet_gpu_ingestible::materialize_table(
  io::scan_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& split = static_cast<parquet_split_info const&>(info);

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  sources.reserve(split.rg_slices.size());
  metadatas.reserve(split.rg_slices.size());
  rg_per_src.reserve(split.rg_slices.size());

  for (auto const& slice : split.rg_slices) {
    if (slice.datasource) {
      sources.push_back(cudf::io::datasource::create(slice.datasource.get()));
    } else {
      // Unclaimed local slice (use_sirius_datasource=false): cudf's bundled
      // datasource wants a plain path, so strip an explicit file:// scheme.
      sources.push_back(cudf::io::datasource::create(strip_file_uri(slice.file_path)));
    }
    metadatas.push_back(*slice.file_metadata);
    rg_per_src.push_back(slice.row_group_indices);
  }
  auto opts = *split.reader_options;
  opts.set_row_groups(std::move(rg_per_src));

  // Per-task AST translation. set_filter is gated on translation success AND on
  // the per-batch disable_filter_pushdown flag (set when the FLBA-decimal probe
  // failed). `sirius_filter_ast` is hoisted so the post-decode fallback can
  // reuse it on a pushdown miss.
  std::unique_ptr<sirius::ast::node> sirius_filter_ast;
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_duckdb_filter_expression) {
    sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    if (!split.disable_filter_pushdown) {
      auto name_resolver = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
        return plan->batch_column_name(ref_index);
      };
      gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
      ast_expression =
        translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
      if (ast_expression) { opts.set_filter(ast_expression->back()); }
    }
  }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  // wdy start
  auto const materialize_start = std::chrono::steady_clock::now();
  // wdy end
  auto [table, _] =
    cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref);
  // wdy start
  auto const materialize_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - materialize_start)
                                         .count();
  // wdy end

  std::optional<fixed_page_spliced_view> fast_spliced_view;
  // wdy start
  bool const can_emit_zero_copy_fixed_page_batch =
    split.fixed_page_reuse && fixed_page_zero_copy_output_enabled() && !_duckdb_filter_expression &&
    !split.needs_assembly;
  // wdy end
  if (split.fixed_page_reuse) {
    if (!_fixed_page_cache || _fixed_page_cache->entry == nullptr) {
      throw std::runtime_error(
        "[fixed-page-cache] split requested hybrid reuse but ingestible has no cache state");
    }
    if (split.fixed_page_reuse->filtered_reuse) {
      if (!sirius_filter_ast || !ast_expression) {
        throw std::runtime_error(
          "[fixed-page-cache] filtered reuse split reached materialize without reader filter");
      }
      table = splice_filtered_fixed_page_cached_columns(std::move(table),
                                                        split,
                                                        *_fixed_page_cache->entry,
                                                        _fixed_page_cache->cached_by_data_index,
                                                        mem_space,
                                                        stream,
                                                        mr_ref,
                                                        *sirius_filter_ast);
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] hybrid_reuse_materialize_filtered pinned={} target_gpu={} "
        "cached_cols={} parquet_cols={} reader_extra_filter_cols={} rows={} columns={}",
        split.fixed_page_reuse->pinned_name,
        mem_space.get_device_id(),
        split.fixed_page_reuse->cached_data_indices.size(),
        split.fixed_page_reuse->parquet_data_indices.size(),
        split.fixed_page_reuse->reader_extra_filter_data_indices.size(),
        table->num_rows(),
        table->num_columns());
    } else {
      if ((sirius_filter_ast && !ast_expression) || can_emit_zero_copy_fixed_page_batch) {
        fast_spliced_view = try_splice_fixed_page_cached_columns_view(
          table, split, *_fixed_page_cache->entry, _fixed_page_cache->cached_by_data_index, stream);
      }
      if (fast_spliced_view) {
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] hybrid_reuse_materialize_view pinned={} target_gpu={} "
          "cached_cols={} parquet_cols={} rows={} columns={}",
          split.fixed_page_reuse->pinned_name,
          mem_space.get_device_id(),
          split.fixed_page_reuse->cached_data_indices.size(),
          split.fixed_page_reuse->parquet_data_indices.size(),
          fast_spliced_view->view.num_rows(),
          fast_spliced_view->view.num_columns());
      } else {
        table = splice_fixed_page_cached_columns(std::move(table),
                                                 split,
                                                 *_fixed_page_cache->entry,
                                                 _fixed_page_cache->cached_by_data_index,
                                                 mem_space,
                                                 stream,
                                                 mr_ref);
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] hybrid_reuse_materialize pinned={} target_gpu={} cached_cols={} "
          "parquet_cols={} rows={} columns={}",
          split.fixed_page_reuse->pinned_name,
          mem_space.get_device_id(),
          split.fixed_page_reuse->cached_data_indices.size(),
          split.fixed_page_reuse->parquet_data_indices.size(),
          table->num_rows(),
          table->num_columns());
      }
    }
  }

  auto const materialized_rows =
    fast_spliced_view ? fast_spliced_view->view.num_rows() : table->num_rows();
  auto const materialized_columns =
    fast_spliced_view ? fast_spliced_view->view.num_columns() : table->num_columns();

  // wdy start
  SIRIUS_LOG_INFO(
    "[scan-audit] parquet_materialize target_gpu={} files={} columns={} row_groups={} "
    "compressed_bytes={} uncompressed_bytes={} column_bytes={} output_rows={} output_columns={} "
    "split_count={} duration_us={}",
    mem_space.get_device_id(),
    scan_audit_file_paths(split.rg_slices),
    join_strings(split.plan->data_column_names(), ','),
    scan_audit_row_groups(split.rg_slices),
    scan_audit_compressed_bytes(split.rg_slices),
    scan_audit_uncompressed_bytes(split.rg_slices),
    scan_audit_column_bytes(split.rg_slices, split.plan->data_column_names()),
    materialized_rows,
    materialized_columns,
    split.rg_slices.size(),
    materialize_duration_us);
  // wdy end

  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::materialize_table] Read {} file(s) (first: {}) — {} rows, {} "
    "columns",
    split.rg_slices.size(),
    split.rg_slices.empty() ? "<none>" : split.rg_slices.front().file_path,
    materialized_rows,
    materialized_columns);

  // wdy start
  if (can_emit_zero_copy_fixed_page_batch && fast_spliced_view) {
    auto batch = make_fixed_page_spliced_view_batch(
      std::move(*fast_spliced_view),
      const_cast<::cucascade::memory::memory_space&>(mem_space),
      stream);
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] zero_copy_output_batch target_gpu={} rows={} columns={}",
      mem_space.get_device_id(),
      materialized_rows,
      materialized_columns);
    return io::filtered_table{nullptr, io::filter_state::UNFILTERED, std::move(batch)};
  }
  // wdy end

  // Determine filter state. When pushdown engaged the reader applied the
  // filter; otherwise we apply post-decode here. `sirius_filter_ast` must
  // outlive `exec` — the executor only borrows the AST.
  io::filter_state state = io::filter_state::UNFILTERED;
  if (sirius_filter_ast) {
    if (!ast_expression) {
      sirius::gpu_expression_executor exec(
        sirius_filter_ast.get(), cudf::get_current_device_resource_ref(), stream);
      // wdy start
      auto const filter_start       = std::chrono::steady_clock::now();
      auto const filter_input_rows  = fast_spliced_view ? fast_spliced_view->view.num_rows()
                                                        : table->num_rows();
      auto const filter_input_cols  = fast_spliced_view ? fast_spliced_view->view.num_columns()
                                                        : table->num_columns();
      bool const filter_input_view  = fast_spliced_view.has_value();
      // wdy end
      if (fast_spliced_view) {
        table = exec.select(fast_spliced_view->view);
        fast_spliced_view.reset();
      } else {
        auto input = std::move(table);
        table      = exec.select(input->view());
      }
      // wdy start
      auto const filter_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - filter_start)
                                        .count();
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] stage_timing stage=post_filter_select target_gpu={} "
        "input_rows={} output_rows={} input_columns={} output_columns={} input_kind={} "
        "duration_us={}",
        mem_space.get_device_id(),
        filter_input_rows,
        table->num_rows(),
        filter_input_cols,
        table->num_columns(),
        filter_input_view ? "spliced_view" : "materialized_table",
        filter_duration_us);
      // wdy end
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible::materialize_table] Applied duckdb filter expression "
        "post-decode.");
    }
    state = io::filter_state::ROW_FILTERED;
  }

  // Reader-side pushdown succeeded and the plan needs assembly — inline it
  // here so the scan operator can skip post_filter_and_project entirely.
  // (post-decode fallback keeps assembly external because re-allocating after
  // exec.select is the same shape either way.)
  if (state == io::filter_state::ROW_FILTERED && ast_expression && split.needs_assembly) {
    // wdy start
    auto const assembly_start = std::chrono::steady_clock::now();
    auto const input_rows     = table->num_rows();
    auto const input_columns  = table->num_columns();
    // wdy end
    table = assemble_scan_output(*_plan, std::move(table), split.partition_values, stream);
    // wdy start
    auto const assembly_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - assembly_start)
                                        .count();
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] stage_timing stage=inline_assembly target_gpu={} input_rows={} "
      "output_rows={} input_columns={} output_columns={} duration_us={}",
      mem_space.get_device_id(),
      input_rows,
      table->num_rows(),
      input_columns,
      table->num_columns(),
      assembly_duration_us);
    // wdy end
    state = io::filter_state::ROW_FILTERED_AND_PROJECTED;
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible::materialize_table] Assembled inline on reader-side pushdown path.");
  }

  return io::filtered_table{std::move(table), state};
}

//===----------------------------------------------------------------------===//
// post_filter_and_project — assembly only
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> parquet_gpu_ingestible::post_filter_and_project(
  std::unique_ptr<cudf::table> input,
  io::post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& /*mem_space*/,
  rmm::cuda_stream_view stream)
{
  auto const& pf = static_cast<parquet_post_filter_and_projection_info const&>(info);
  // The per-batch assembly call. The ingestible only emits a non-null
  // post_filter_and_projection_info when needs_output_assembly(*_plan) is true,
  // so this is unconditionally meaningful.
  // wdy start
  auto const assembly_start = std::chrono::steady_clock::now();
  auto const input_rows     = input->num_rows();
  auto const input_columns  = input->num_columns();
  // wdy end
  auto out = assemble_scan_output(*_plan, std::move(input), pf.partition_values, stream);
  // wdy start
  auto const assembly_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - assembly_start)
                                      .count();
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] stage_timing stage=post_filter_project_assembly input_rows={} "
    "output_rows={} input_columns={} output_columns={} duration_us={}",
    input_rows,
    out->num_rows(),
    input_columns,
    out->num_columns(),
    assembly_duration_us);
  // wdy end
  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::post_filter_and_project] Assembled scan output to plan layout.");
  return out;
}

}  // namespace sirius::op::scan
