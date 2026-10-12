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
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/cache_hit_metrics.hpp"
#include "op/scan/owning_table_view.hpp"

#include <expression/ast/from_duckdb.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <io/io_context.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_metadata.hpp>
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_dynamic_filter.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// duckdb
#include <duckdb/common/hive_partitioning.hpp>

// uring_reactor MUST be included last among sirius headers — see
// parquet_split_provider.cpp for the BLOCK_SIZE macro-collision rationale.
#include <io/uring/uring_reactor.hpp>

// standard library
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

bool has_uri_scheme(std::string const& p) { return p.find("://") != std::string::npos; }

std::string join_strings(std::vector<std::string> const& values, char delimiter)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) { out << delimiter; }
    out << values[i];
  }
  return out.str();
}

std::string scan_audit_row_groups(std::vector<row_group_slice> const& slices)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < slices.size(); ++i) {
    if (i != 0) { out << '|'; }
    if (slices[i].row_group_indices.empty()) {
      out << '-';
      continue;
    }
    for (std::size_t j = 0; j < slices[i].row_group_indices.size(); ++j) {
      if (j != 0) { out << ','; }
      out << slices[i].row_group_indices[j];
    }
  }
  return out.str();
}

bool is_fixed_width_auto_cache_candidate(cudf::column_view const& col) noexcept
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

bool fixed_page_auto_cache_enabled()
{
  auto const* reuse = std::getenv("SIRIUS_ENABLE_FIXED_PAGE_REUSE");
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_AUTO_CACHE");
  return reuse != nullptr && std::string_view(reuse) == "1" && value != nullptr &&
         std::string_view(value) == "1";
}

bool fixed_page_owned_pages_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_OWNED_PAGES");
  return value != nullptr && std::string_view(value) == "1";
}

bool fixed_page_direct_auto_populate_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_DIRECT_AUTO_POPULATE");
  return value == nullptr || std::string_view(value) != "0";
}

/// Also populate the pinned-entry store for a batch the flat page store has already taken.
/// Off by default: every batch was being deep-copied into both stores, and the pinned one
/// served 0 of 123 reuse attempts on ClickBench and 3 of 252 on TPC-H SF50 while holding
/// 1.47GiB on average outside the eviction the page store gets. Set to 1 to A/B the old
/// behaviour. Batches the page store cannot take (multi-slice) still go to the pinned store.
bool pinned_entry_cache_for_paged_batches_enabled()
{
  auto const* value = std::getenv("SIRIUS_PINNED_ENTRY_CACHE");
  return value != nullptr && std::string_view(value) == "1";
}

/// Let string-only batches into the page store (see auto_cache_materialized_table). On by
/// default; set to 0 to refuse them as before.
bool page_store_string_only_enabled()
{
  auto const* value = std::getenv("SIRIUS_PAGE_STORE_STRING_ONLY");
  return value == nullptr || std::string_view(value) != "0";
}

/// Serve the pages a column has and read only the rows it lacks (HANDOFF B4). On by default;
/// SIRIUS_PAGE_PARTIAL_SERVE=0 falls back to reading such a column whole.
bool page_partial_serve_enabled()
{
  auto const* value = std::getenv("SIRIUS_PAGE_PARTIAL_SERVE");
  return value == nullptr || std::string_view(value) != "0";
}

using cache_hit_metrics::cache_hit_trace_enabled;
using cache_hit_metrics::decoded_table_bytes;

bool scan_prep_sync_enabled()
{
  static bool const on = [] {
    auto const* value = std::getenv("SIRIUS_SCAN_PREP_SYNC");
    return value != nullptr && std::string_view(value) == "1";
  }();
  return on;
}

bool scan_cache_trace_enabled()
{
  static bool const on = [] {
    auto const* v = std::getenv("SIRIUS_SCAN_CACHE_TRACE");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

bool fixed_page_hybrid_provider_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_HYBRID_PROVIDER");
  return value != nullptr && std::string_view(value) == "1";
}

/// Stamp the cache entry's filter identity only when the reader actually applied a
/// filter. Default on; set SIRIUS_FIXED_PAGE_HONEST_FILTER_SIGNATURE=0 to restore the
/// previous behaviour of stamping unconditionally, which labels unfiltered tables as
/// filtered whenever AST translation rejected the predicate.
/// Cache the table BEFORE the reader's filter is applied, so an entry is valid for
/// any query over the same columns rather than only for the one predicate it was
/// built with. Forces reader-side pushdown off; post_filter_and_project then
/// applies each consuming query's predicate post-decode, which the cached-batch
/// path already does (batches leave as filter_state::UNFILTERED).
///
/// The trade is I/O and memory: the reader stops skipping rows, so every scan
/// reads its columns whole and the cached entry is the full column, not a subset.
/// Cache the rows of a cacheable scan before the predicate and filter after, so the cached
/// pages serve every later query over those row groups, filtered or not. On by default:
/// measured on ClickBench it took the bytes served from cache from 5.5% to 14.5% and the run
/// from -2.1% to -8.0% against baseline. SIRIUS_FIXED_PAGE_CACHE_BEFORE_FILTER=0 turns it off.
bool cache_before_filter_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_BEFORE_FILTER");
  return value == nullptr || std::string_view(value) != "0";
}

/// Whether a scan whose rows the READER already filtered may populate the cache.
///
/// Such a scan produces an entry holding "the rows matching this predicate", so it
/// carries a filter signature and can only ever serve a scan with the byte-identical
/// predicate -- in practice only a later execution of the same query. Measured on
/// SF100: 84 of 111 populates land in such entries (ClickBench: 27 of 32), and
/// lineitem alone accumulates 11 distinct filter entries against 12 unfiltered
/// scans that could all share one. The budget those 11 hold is why the unfiltered
/// entry stops widening and later scans miss with `missing_columns` (SF100 102,
/// ClickBench 460).
///
/// Skipping them concentrates the budget on the unfiltered entry, which is a
/// superset of every predicate and therefore reusable by all of them. The cost is
/// the same-query repeat hits those entries did provide.
bool cache_filtered_scans_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_FILTERED_SCANS");
  return value == nullptr || std::string_view(value) != "0";
}

/// Drop the predicate from the cache key and rely on per-chunk value ranges.
bool filter_keyless_cache_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_FILTER_KEYLESS_CACHE");
  return value != nullptr && std::string_view(value) == "1";
}

bool honest_filter_signature_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_HONEST_FILTER_SIGNATURE");
  return value == nullptr || std::string_view(value) != "0";
}

std::string auto_fixed_page_cache_name(std::vector<std::string> const& file_paths,
                                       std::string const& filter_signature,
                                       std::vector<std::string> const& column_names)
{
  std::vector<std::string> paths = file_paths;
  std::sort(paths.begin(), paths.end());
  std::ostringstream out;
  out << "__wdy_auto_fixed_page:";
  for (auto const& path : paths) { out << path << ";"; }
  // Keyless mode: the predicate stops being part of the cache IDENTITY and becomes
  // metadata on each chunk (chunk_provenance::filter_ranges). One entry per file
  // then holds chunks filtered differently, and a consumer takes the chunks whose
  // predicate its own is narrower than, instead of only an entry whose predicate
  // text matches byte for byte.
  if (!filter_signature.empty() && !filter_keyless_cache_enabled()) {
    out << "filter=" << filter_signature;
  }
  // Column set is part of the cache identity: two queries over the same
  // (file, filter) that project different columns are genuinely different
  // cache entries. Folding the column set into the key -- instead of just
  // (file, filter) -- lets both projections stay resident side by side
  // instead of the second one's insert finding a name collision, failing
  // the "same column set" appendable check, and erasing the first
  // projection's fully-cached data to rebuild from scratch for its own
  // columns (see insert_fixed_page_entry_from_view's `appendable` branch).
  // Column-keyed mode drops the column set from the identity so every projection
  // over the same (file, filter) shares ONE entry, and inserts merge into it
  // instead of colliding. Measured on ClickBench: SearchPhrase is read by 13 of
  // 37 queries but, split across three projection keys, no key was reused more
  // than 3 times -- and each key stored its own 1.07 GB copy. Merging turns both
  // halves of that around at once. Requires the merge path below to key chunks by
  // their row groups rather than by arrival order, which is what
  // chunk_provenance_by_index records.
  if (scan_manager::sirius_scan_manager::column_keyed_cache_is_enabled()) { return out.str(); }
  std::vector<std::string> cols = column_names;
  std::sort(cols.begin(), cols.end());
  out << ";cols=";
  for (auto const& col : cols) { out << col << ","; }
  return out.str();
}

//===----------------------------------------------------------------------===//
// parquet_batch_coalescer
//===----------------------------------------------------------------------===//
/**
 * @brief Coalesces per-file metadata units into data-batch splits.
 *
 * Receives one @c parquet_file_scan_info per file (each already pruned and
 * byte-accounted by the metadata-scan task) and accumulates their row groups
 * into @c parquet_split_info batches sized to @c approximate_batch_size. A
 * single large file spans multiple splits (each with its own row_group_slice),
 * and several small files bundle into one split. Bundling across files is only
 * safe when they share hive-partition values and the same pushdown decision, so
 * a mismatch on either forces a flush.
 */
class parquet_batch_coalescer : public batch_coalescer {
 public:
  parquet_batch_coalescer(std::size_t cap,
                          std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
                          std::shared_ptr<scan_plan const> plan)
    : _cap(cap),
      _reader_options(std::move(reader_options)),
      _plan(std::move(plan)),
      _needs_assembly(needs_output_assembly(*_plan))
  {
  }

  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    std::vector<std::unique_ptr<scan_info>> emitted;
    auto* file = dynamic_cast<parquet_file_scan_info*>(info.get());
    if (file == nullptr) { return emitted; }

    // Remember the first fully-pruned file. If the WHOLE source coalesces to
    // nothing, flush() emits one empty split built from it — zero splits mean
    // zero tasks, and the pipeline-completion accounting only fires from task
    // completion, hanging sirius_engine::execute().
    if (file->row_groups.empty() && !_empty_split_fallback) {
      _empty_split_fallback = fallback_file{
        file->file_metadata,
        file->file_path,
        file->datasource ? std::shared_ptr<io::sirius_datasource>(file->datasource->duplicate())
                         : std::shared_ptr<io::sirius_datasource>{},
        file->partition_values,
        file->disable_filter_pushdown};
    }

    if (!_slices.empty() && (_partition_values != file->partition_values ||
                             _disable_pushdown != file->disable_filter_pushdown)) {
      emitted.push_back(emit_current());
    }
    _partition_values = file->partition_values;
    _disable_pushdown = file->disable_filter_pushdown;

    std::vector<cudf::size_type> cur_rgs;
    std::size_t cur_unc  = 0;
    std::size_t cur_comp = 0;
    int64_t cur_rows     = 0;
    auto seal_file       = [&]() {
      if (cur_rgs.empty()) { return; }
      // A file's row groups can span multiple splits, each sealed into its own
      // slice. fadvise stores a per-scan prefetch handle on the datasource, so
      // each slice gets its own datasource (sharing the io_object) — otherwise
      // a later split's fadvise would stomp an earlier one's handle.
      auto slice_ds = file->datasource
                              ? std::shared_ptr<io::sirius_datasource>(file->datasource->duplicate())
                              : std::shared_ptr<io::sirius_datasource>{};
      _slices.emplace_back(file->file_metadata,
                           file->file_path,
                           std::move(cur_rgs),
                           cur_unc,
                           cur_comp,
                           std::move(slice_ds));
      _produced_any = true;
      _acc_bytes += cur_unc;
      _acc_rows += cur_rows;
      cur_rgs.clear();
      cur_unc  = 0;
      cur_comp = 0;
      cur_rows = 0;
    };

    // cuDF tables are limited to cudf::size_type (int32_t) rows per call.
    static constexpr int64_t cudf_max_rows = std::numeric_limits<cudf::size_type>::max();

    for (auto const& rg : file->row_groups) {
      bool const byte_cap_hit = (!_slices.empty() || !cur_rgs.empty()) && _cap > 0 &&
                                _acc_bytes + cur_unc + rg.uncompressed_bytes > _cap;
      bool const row_cap_hit = (!_slices.empty() || !cur_rgs.empty()) &&
                               _acc_rows + cur_rows + rg.num_rows > cudf_max_rows;
      if (byte_cap_hit || row_cap_hit) {
        seal_file();
        emitted.push_back(emit_current());
      }
      cur_unc += rg.uncompressed_bytes;
      cur_comp += rg.compressed_bytes;
      cur_rgs.push_back(rg.index);
      cur_rows += rg.num_rows;
    }
    seal_file();
    return emitted;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (!_slices.empty()) { out.push_back(emit_current()); }
    // Every file was stats-pruned to zero row groups: emit exactly one split
    // with a single zero-row-group slice so the scan still creates one task
    // (materialize_metadata_to_table short-circuits it to a schema-correct
    // empty table). Partial prunes never reach here — any surviving slice sets
    // _produced_any. Zero splits would mean zero tasks, and pipeline-completion
    // accounting only fires from task completion, hanging the query.
    if (!_produced_any && _empty_split_fallback) {
      _slices.emplace_back(_empty_split_fallback->file_metadata,
                           _empty_split_fallback->file_path,
                           std::vector<cudf::size_type>{},
                           /*reserved_uncompressed_bytes=*/0,
                           /*reserved_compressed_bytes=*/0,
                           _empty_split_fallback->datasource);
      _partition_values = _empty_split_fallback->partition_values;
      _disable_pushdown = _empty_split_fallback->disable_filter_pushdown;
      _produced_any     = true;
      out.push_back(emit_current());
    }
    return out;
  }

 private:
  std::unique_ptr<scan_info> emit_current()
  {
    auto split                     = std::make_unique<parquet_split_info>();
    split->rg_slices               = std::move(_slices);
    split->reader_options          = _reader_options;
    split->plan                    = _plan;
    split->disable_filter_pushdown = _disable_pushdown;
    split->needs_assembly          = _needs_assembly;
    split->partition_values        = _partition_values;
    SIRIUS_LOG_INFO(
      "[coalesce-debug] parquet_batch_coalescer emit #{}: {} slice(s), acc_bytes={}, cap={}",
      ++_emit_count,
      split->rg_slices.size(),
      _acc_bytes,
      _cap);
    _slices.clear();
    _acc_bytes = 0;
    _acc_rows  = 0;
    return split;
  }

  const std::size_t _cap;
  std::shared_ptr<cudf::io::parquet_reader_options> _reader_options;
  std::shared_ptr<scan_plan const> _plan;
  const bool _needs_assembly;

  std::vector<row_group_slice> _slices;
  std::size_t _acc_bytes  = 0;
  int64_t _acc_rows       = 0;
  std::size_t _emit_count = 0;  // [coalesce-debug] running count of emitted batches
  std::vector<std::string> _partition_values;
  bool _disable_pushdown = false;

  /// First fully-pruned file, kept as the source for flush()'s empty-split
  /// fallback when the whole scan produced no slice.
  struct fallback_file {
    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    std::string file_path;
    std::shared_ptr<io::sirius_datasource> datasource;
    std::vector<std::string> partition_values;
    bool disable_filter_pushdown;
  };
  std::optional<fallback_file> _empty_split_fallback;
  bool _produced_any = false;
};

/// Column-chunk byte ranges a read fetches for @p row_group_indices, honoring
/// @p options' column projection — the ranges materialize_table reads, used to
/// drive prefetch. Empty when there are no row groups.
std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& row_group_indices)
{
  if (row_group_indices.empty()) { return {}; }
  hybrid_scan_reader reader(metadata, options);
  return reader.all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size()),
    options);
}

}  // namespace

//===----------------------------------------------------------------------===//
// scan_info fadvise_entries — prefetch byte ranges
//===----------------------------------------------------------------------===//
std::vector<scan_info::fadvise_entry> parquet_file_scan_info::fadvise_entries() const
{
  if (!datasource || !file_metadata || !reader_options) { return {}; }
  std::vector<cudf::size_type> rg_indices;
  rg_indices.reserve(row_groups.size());
  for (auto const& rg : row_groups) {
    rg_indices.push_back(rg.index);
  }
  auto ranges = column_chunk_ranges(*file_metadata, *reader_options, rg_indices);
  if (ranges.empty()) { return {}; }
  fadvise_entry entry;
  entry.datasource = datasource;
  entry.ranges     = std::move(ranges);
  std::vector<fadvise_entry> out;
  out.push_back(std::move(entry));
  return out;
}

std::vector<scan_info::fadvise_entry> parquet_split_info::fadvise_entries() const
{
  if (!reader_options) { return {}; }
  std::vector<fadvise_entry> entries;
  entries.reserve(rg_slices.size());
  for (auto const& slice : rg_slices) {
    if (!slice.datasource || !slice.file_metadata) { continue; }
    auto ranges =
      column_chunk_ranges(*slice.file_metadata, *reader_options, slice.row_group_indices);
    if (ranges.empty()) { continue; }
    fadvise_entry entry;
    entry.datasource = slice.datasource;
    entry.ranges     = std::move(ranges);
    entries.push_back(std::move(entry));
  }
  return entries;
}

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<parquet_gpu_ingestible> make_ingestible(
  std::unique_ptr<parquet_ingestible_table_info> info)
{
  return std::make_shared<parquet_gpu_ingestible>(std::move(info));
}

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
parquet_gpu_ingestible::parquet_gpu_ingestible(std::unique_ptr<parquet_ingestible_table_info> info)
  : _info(std::move(info))
{
  auto const& bind = static_cast<parquet_ingestible_table_info const&>(table_info());

  // Any non-trivial scan shape — reader-side projection (incl. a pruned/reordered
  // column_ids with empty projection_ids, the no-pushdown sirius_read_parquet
  // case), filter pushdown, or hive-partition injection — needs column names.
  // Matches parquet_split_provider's ctor invariant and build_scan_plan's
  // needs_reader_projection trigger.
  bool const needs_names = !bind.projection_ids.empty() ||
                           (bind.table_filters && !bind.table_filters->filters.empty()) ||
                           !bind.partition_indices.empty() ||
                           column_ids_need_reader_projection(bind.column_ids, bind.names.size());
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

  // Shared reader options — column projection only. set_filter is never applied
  // here: it is a per-split decision (FLBA files disable it) made in
  // materialize_table on a copy of these options.
  _reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  if (_plan->is_projected()) { _reader_options->set_column_names(_plan->data_column_names()); }

  _sirius_dynamic_filters = bind.sirius_dynamic_filters;

  // Producers reference probe columns in DuckDB's column_ids space; the AST merge and the
  // post-decode apply both key by output-column position. Install the translation so push_filter
  // remaps before storing. Wiring-time setup, before the producing build publishes.
  if (_sirius_dynamic_filters) {
    _sirius_dynamic_filters->set_consumer_column_remap(_plan->output_position_by_column_id);
  }

  // Hive-partition columns are path-derived constants, not decoded parquet columns, so they must
  // not receive post-decode dynamic filters.
  if (_sirius_dynamic_filters && _plan->has_partitions()) {
    std::vector<std::size_t> partition_cols;
    for (std::size_t i = 0; i < _plan->output_layout.size(); ++i) {
      if (_plan->output_layout[i].source == scan_plan::output_entry::PARTITION) {
        partition_cols.push_back(i);
      }
    }
    _sirius_dynamic_filters->ignore_columns(partition_cols);
  }

  _file_paths = bind.resolved_file_paths;
}

parquet_gpu_ingestible::~parquet_gpu_ingestible() = default;

bool parquet_gpu_ingestible::waits_for_row_id_filter() const
{
  return _plan->file_row_number_output_position().has_value() && _sirius_dynamic_filters &&
         _sirius_dynamic_filters->has_producers();
}

filtered_table parquet_gpu_ingestible::read_row_numbers(parquet_split_info const& split,
                                                        std::vector<std::int64_t> const& rows,
                                                        cucascade::memory::memory_space const& mem_space,
                                                        rmm::cuda_stream_view stream)
{
  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  // Rows closer than this are read together: one read call per run, not per row.
  constexpr std::int64_t kMergeGap = 4096;
  auto finish = [&](std::unique_ptr<cudf::table> table) {
    if (cache_hit_trace_enabled()) {
      // The appended row/file identities are generated, not supplied by either cache or reader.
      auto const data_count = table->num_columns() -
                              static_cast<cudf::size_type>(_plan->virtual_columns.size());
      std::vector<cudf::size_type> data_indices(static_cast<std::size_t>(data_count));
      std::iota(data_indices.begin(), data_indices.end(), 0);
      auto const total = decoded_table_bytes(table->view().select(data_indices), stream);
      SIRIUS_LOG_INFO("[cache-hit] file='{}' total_bytes={} read_bytes={} kind=miss "
                      "path=late_refetch rows={} metric=value_bytes_v2",
                      split.rg_slices.front().file_path, total, total, table->num_rows());
    }
    return filtered_table{owning_table_view{std::move(table)}, filter_state::UNFILTERED};
  };
  std::vector<std::unique_ptr<cudf::table>> parts;
  std::size_t ranges = 0, read_rows = 0;
  for (auto const& slice : split.rg_slices) {
    if (!slice.file_metadata) { continue; }
    auto const& rgs = slice.file_metadata->row_groups;
    std::vector<std::int64_t> start(rgs.size() + 1, 0);
    for (std::size_t i = 0; i < rgs.size(); ++i) {
      start[i + 1] = start[i] + static_cast<std::int64_t>(rgs[i].num_rows);
    }
    auto const it = std::find(_file_paths.begin(), _file_paths.end(), slice.file_path);
    auto const file_index =
      static_cast<std::uint64_t>(it == _file_paths.end() ? 0 : it - _file_paths.begin());
    // Runs of wanted rows inside this split's row groups.
    std::vector<std::pair<std::int64_t, std::int64_t>> runs;  // [begin, end)
    for (auto const rg : slice.row_group_indices) {
      auto const b = start[static_cast<std::size_t>(rg)];
      auto const e = start[static_cast<std::size_t>(rg) + 1];
      for (auto k = std::lower_bound(rows.begin(), rows.end(), b); k != rows.end() && *k < e; ++k) {
        if (!runs.empty() && *k - runs.back().second <= kMergeGap && *k >= runs.back().first &&
            runs.back().second <= e && runs.back().first >= b) {
          runs.back().second = *k + 1;
        } else {
          runs.emplace_back(*k, *k + 1);
        }
      }
    }
    for (auto const& [rb, re] : runs) {
      auto o = *split.reader_options;
      o.set_skip_rows(rb);
      o.set_num_rows(re - rb);
      std::vector<std::unique_ptr<cudf::io::datasource>> src;
      src.push_back(slice.datasource ? cudf::io::datasource::create(slice.datasource.get())
                                     : cudf::io::datasource::create(slice.file_path));
      std::vector<cudf::io::parquet::FileMetaData> md{*slice.file_metadata};
      auto tbl  = cudf::io::read_parquet(std::move(src), std::move(md), o, stream, mr_ref).tbl;
      auto cols = tbl->release();
      for (auto const& vc : _plan->virtual_columns) {
        if (vc.kind == scan_plan::virtual_column::FILE_ROW_NUMBER) {
          cudf::numeric_scalar<std::int64_t> init(rb, true, stream);
          cudf::numeric_scalar<std::int64_t> step(1, true, stream);
          cols.push_back(cudf::sequence(static_cast<cudf::size_type>(re - rb), init, step, stream, mr_ref));
        } else {
          cudf::numeric_scalar<std::uint64_t> fi(file_index, true, stream);
          cols.push_back(cudf::make_column_from_scalar(fi, static_cast<cudf::size_type>(re - rb), stream, mr_ref));
        }
      }
      parts.push_back(std::make_unique<cudf::table>(std::move(cols)));
      ++ranges;
      read_rows += static_cast<std::size_t>(re - rb);
    }
  }
  SIRIUS_LOG_INFO("[late-materialization] read {} rows in {} runs for {} wanted row numbers",
                  read_rows, ranges, rows.size());
  if (parts.empty()) {
    // Nothing of this split is wanted: an empty batch with the right schema (zero-row read plus
    // empty virtual columns).
    auto o = *split.reader_options;
    o.set_num_rows(0);
    std::vector<std::unique_ptr<cudf::io::datasource>> src;
    auto const& slice = split.rg_slices.front();
    src.push_back(slice.datasource ? cudf::io::datasource::create(slice.datasource.get())
                                   : cudf::io::datasource::create(slice.file_path));
    std::vector<cudf::io::parquet::FileMetaData> md{*slice.file_metadata};
    auto cols = cudf::io::read_parquet(std::move(src), std::move(md), o, stream, mr_ref).tbl->release();
    for (auto const& vc : _plan->virtual_columns) {
      cols.push_back(cudf::make_empty_column(vc.kind == scan_plan::virtual_column::FILE_ROW_NUMBER
                                               ? cudf::data_type{cudf::type_id::INT64}
                                               : cudf::data_type{cudf::type_id::UINT64}));
    }
    return finish(std::make_unique<cudf::table>(std::move(cols)));
  }
  if (parts.size() == 1) {
    return finish(std::move(parts.front()));
  }
  std::vector<cudf::table_view> views;
  for (auto const& p : parts) { views.push_back(p->view()); }
  return finish(cudf::concatenate(views, stream, mr_ref));
}

bool parquet_gpu_ingestible::row_id_filter_published() const
{
  auto const pos = _plan->file_row_number_output_position();
  return pos && _sirius_dynamic_filters &&
         !_sirius_dynamic_filters->filters_for_column(*pos).empty();
}

//===----------------------------------------------------------------------===//
// coalescer / post-filter factories
//===----------------------------------------------------------------------===//
std::unique_ptr<batch_coalescer> parquet_gpu_ingestible::create_batch_coalescer() const
{
  return std::make_unique<parquet_batch_coalescer>(
    _info->approximate_batch_size, _reader_options, _plan);
}

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool parquet_gpu_ingestible::has_processed_all_metadata() const
{
  return _next_file_idx.load(std::memory_order_relaxed) >= _file_paths.size();
}

std::function<std::unique_ptr<op::scan::scan_info>()> parquet_gpu_ingestible::next_split_provider(
  io::ioctx_resolver resolve)
{
  if (!resolve) { throw std::runtime_error("parquet_gpu_ingestible: no scan_manager is wired."); }
  auto const idx = _next_file_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _file_paths.size()) { return nullptr; }  // lost the race for the final file

  // Route each file to its own backend (s3:// -> rest, local -> uring/kvikio) so a
  // mixed-scheme scan opens every file on the right ioctx.  One metadata-scan task
  // per file; row-group chunking and file bundling happen downstream in
  // parquet_batch_coalescer.
  auto const& file_path = _file_paths[idx];
  // The resolver returns a valid ioctx or throws if no backend supports the path.
  auto io_ctx = resolve(file_path);
  return [this, file_path, io_ctx = std::move(io_ctx)]() -> std::unique_ptr<scan_info> {
    return build_file_scan_info(file_path, io_ctx);
  };
}

//===----------------------------------------------------------------------===//
// build_file_scan_info — per-file footer read + row-group pruning
//===----------------------------------------------------------------------===//
std::unique_ptr<scan_info> parquet_gpu_ingestible::build_file_scan_info(
  std::string const& file_path, std::shared_ptr<io::sirius_ioctx> const& io_ctx)
{
  auto stream = cudf::get_default_stream();

  // Resolve the file to a sirius_datasource (own io backend, prefetch cache and
  // cached metadata). Fall back to a plain cudf datasource only for local paths
  // no sirius backend claims.
  std::shared_ptr<io::sirius_datasource> sirius_ds = io_ctx->open_datasource(file_path);
  if (!sirius_ds && has_uri_scheme(file_path)) {
    throw std::runtime_error("[parquet_gpu_ingestible] no backend supports path: " + file_path);
  }

  // Local copy of the shared options; the per-file filter pushdown decision is
  // applied here, never on _reader_options.
  auto opts = *_reader_options;

  // Obtain footer metadata — from the datasource's cached parquet_metadata when
  // present, else by fetching and parsing the footer.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  if (sirius_ds) {
    if (auto cached = sirius_ds->metadata()) {
      if (auto pm = std::dynamic_pointer_cast<parquet_metadata>(std::move(cached))) {
        file_metadata = pm->file_metadata();
      }
    }
  }
  if (!file_metadata) {
    auto footer           = cudf::io::parquet::fetch_footer_to_host(*sirius_ds);
    auto const footer_len = footer->size();
    hybrid_scan_reader footer_reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                                     opts);
    file_metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(footer_reader.parquet_metadata());
    // Park the parse in the ioctx metadata store so a later scan of the same
    // file skips the footer fetch + Thrift parse (the read above already
    // dereferences *sirius_ds, so it is non-null here). Best-effort.
    [[maybe_unused]] auto const stored =
      sirius_ds->store_metadata(std::make_shared<parquet_metadata>(file_metadata, footer_len));
  }
  auto const& metadata = *file_metadata;

  // FLBA-decimal pushdown probe: cudf's row-group stats filter cannot compare a
  // fixed_point_scalar AST literal against FLBA / BYTE_ARRAY decimal stats, so
  // reader-side pushdown is disabled when such a decimal is among the columns
  // this scan reads (the filter still applies post-decode).
  bool const restrict_to_scanned = _plan->is_projected();
  std::unordered_set<std::string> scanned_column_names;
  if (restrict_to_scanned) {
    auto const names = _plan->data_column_names();
    scanned_column_names.insert(names.begin(), names.end());
  }
  // Two independent decisions that used to share one flag.
  //
  // Row-group pruning reads footer statistics and nothing else, so it is free and
  // always worth doing. Row filtering happens during decode and is what makes a
  // cached page hold "the rows matching this predicate" rather than a contiguous
  // row range of the table -- which is why such a page needs a filter identity and
  // can only ever serve a byte-identical predicate.
  //
  // Caching before the filter therefore wants pruning ON and row filtering OFF.
  // Folding both into one flag meant turning off row filtering also turned off
  // pruning, so the reader read every row group the predicate would have skipped:
  // measured on ClickBench, 40.5 s -> 73.7 s, i.e. below its own no-cache
  // baseline. That measurement says nothing about pre-filter caching itself.
  //
  // Scoped to scans that can actually be cached: a scan carrying dynamic filters is
  // refused by auto_cache_materialized_table regardless, so turning its row filter
  // off would forfeit the reader's pruning and buy nothing.
  bool const cache_before_filter = cache_before_filter_enabled() &&
                                   fixed_page_auto_cache_enabled() &&
                                   !fixed_page_cache_has_dynamic_filters();
  // FLBA-decimal probe: cudf's row-group stats filter cannot compare a
  // fixed_point_scalar AST literal against FLBA / BYTE_ARRAY decimal stats, so
  // such a file must have BOTH off (the filter still applies post-decode).
  bool decimal_blocks_pushdown = false;
  for (auto const& elem : metadata.schema) {
    if (restrict_to_scanned && !scanned_column_names.contains(elem.name)) { continue; }
    bool const is_decimal = (elem.converted_type.has_value() &&
                             *elem.converted_type == cudf::io::parquet::ConvertedType::DECIMAL) ||
                            (elem.logical_type.has_value() &&
                             elem.logical_type->type == cudf::io::parquet::LogicalType::DECIMAL);
    if (!is_decimal) { continue; }
    if (elem.type == cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY ||
        elem.type == cudf::io::parquet::Type::BYTE_ARRAY) {
      decimal_blocks_pushdown = true;
      break;
    }
  }
  bool const prune_row_groups = !decimal_blocks_pushdown;
  // Row-identity virtual columns are numbered from the rows of each row group as read, so the
  // reader must not drop any: the filter runs post-decode instead.
  bool const apply_row_filter =
    !decimal_blocks_pushdown && !cache_before_filter && !_plan->has_virtual_columns();

  // Translate the filter for reader-side row-group pruning unless disabled. The
  // translated cuDF AST must outlive filter_row_groups_with_stats below.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  // Translated for PRUNING even when the row filter is off: filter_row_groups_with_stats
  // reads the predicate off these options.
  if (_duckdb_filter_expression && prune_row_groups) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { opts.set_filter(ast_expression->back()); }
  }

  hybrid_scan_reader reader(metadata, opts);

  // Per-file leaf-column selection for byte accounting. Pure-filter columns are
  // read for filter evaluation but excluded from the uncompressed accounting.

  // DuckDB schema types (P-space), indexed by scan_plan::data_column::primary_idx.
  // Used below to estimate the decoded (GPU-resident) byte size of each projected
  // column when partitioning row groups into batches — see rg_contribution.
  auto const& returned_types   = _info->returned_types;
  auto const data_column_names = _plan->data_column_names();
  std::vector<std::size_t> selected_chunk_indices;
  // Parallel to selected_chunk_indices: the decoded (GPU) byte width of each
  // selected leaf chunk's column, or 0 for VARCHAR / nested / unknown types
  // (which fall back to the parquet encoded-uncompressed size in rg_contribution).
  std::vector<std::size_t> selected_chunk_decoded_width;
  // Parallel to selected_chunk_indices: which data column (index into
  // data_column_names) each selected leaf chunk belongs to, so the cache-entry
  // estimate below can be accumulated per column rather than only in total.
  std::vector<std::size_t> selected_chunk_column_idx;
  std::unordered_set<std::size_t> pure_filter_chunk_indices;
  if (_plan->is_projected()) {
    auto const pure_filter_positions = _plan->pure_filter_batch_positions();
    selected_chunk_indices.reserve(data_column_names.size());
    selected_chunk_decoded_width.reserve(data_column_names.size());
    for (std::size_t k = 0; k < data_column_names.size(); ++k) {
      auto leaves = detail::leaf_indices_for_column(metadata, data_column_names[k]);
      if (leaves.empty()) {
        throw std::runtime_error("[parquet_gpu_ingestible] Projected column '" +
                                 data_column_names[k] +
                                 "' not found in parquet file: " + file_path);
      }
      // Decoded byte width for this data column: fixed-width types use their
      // cuDF decoded width; VARCHAR (fixed_width_byte_size()==0) and nested
      // types (which throw) get 0, signalling rg_contribution to fall back to
      // the encoded-uncompressed byte size.
      std::size_t decoded_width = 0;
      if (k < _plan->data_columns.size()) {
        auto const primary_idx = _plan->data_columns[k].primary_idx;
        if (primary_idx < returned_types.size()) {
          try {
            decoded_width = returned_types[primary_idx].fixed_width_byte_size();
          } catch (...) {
            decoded_width = 0;  // VARCHAR/LIST/STRUCT/etc — fall back to encoded size
          }
        }
      }
      bool const is_pure_filter = pure_filter_positions.count(k);
      for (auto const leaf : leaves) {
        selected_chunk_indices.push_back(leaf);
        selected_chunk_decoded_width.push_back(decoded_width);
        selected_chunk_column_idx.push_back(k);
        if (is_pure_filter) { pure_filter_chunk_indices.insert(leaf); }
      }
    }
  }

  auto row_group_indices = reader.all_row_groups(opts);
  // Drop what the page cache is serving. Done before stats pruning so the two
  // compose: pruning removes row groups the predicate cannot match, this removes
  // ones already resident, and what survives is exactly the residual to read.
  if (!_cached_row_groups.empty()) {
    auto const cached = _cached_row_groups.find(file_path);
    if (cached != _cached_row_groups.end() && !cached->second.empty()) {
      auto const before = row_group_indices.size();
      std::erase_if(row_group_indices, [&](auto rg) {
        return cached->second.contains(static_cast<int>(rg));
      });
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] residual_scan file='{}' row_groups {} -> {} (cache serves {})",
        file_path,
        before,
        row_group_indices.size(),
        cached->second.size());
    }
  }
  if (ast_expression && prune_row_groups) {
    auto const rgs_before = row_group_indices.size();
    row_group_indices     = reader.filter_row_groups_with_stats(row_group_indices, opts, stream);
    SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] Row group pruning {}: {} -> {} row group(s)",
                     file_path,
                     rgs_before,
                     row_group_indices.size());
  }

  // Estimate the DECODED (GPU-resident) byte size of a row group's projected
  // columns
  auto rg_contribution = [&](cudf::io::parquet::RowGroup const& row_group) {
    std::size_t rg_decoded    = 0;
    std::size_t rg_compressed = 0;
    auto const row_count      = static_cast<std::size_t>(row_group.num_rows);
    auto add_chunk            = [&](cudf::io::parquet::ColumnChunk const& chunk,
                         bool is_pure_filter,
                         std::size_t decoded_width) {
      auto const& column_metadata = chunk.meta_data;
      if (!is_pure_filter) {
        if (decoded_width > 0) {
          // Fixed-width column: row_count x decoded width, plus a validity mask.
          rg_decoded += row_count * decoded_width + row_count / 8;
        } else {
          // VARCHAR / nested / unknown. Dictionary/RLE encoding can make the
          // encoded chunk many times smaller than its decoded char buffer, so
          // prefer SizeStatistics::unencoded_byte_array_data_bytes (the exact
          // decoded BYTE_ARRAY size) when the writer recorded it, else fall back
          // to the encoded-uncompressed size (under-counts dictionary data).
          std::size_t const char_bytes =
            (column_metadata.size_statistics &&
             column_metadata.size_statistics->unencoded_byte_array_data_bytes)
                         ? static_cast<std::size_t>(
                  *column_metadata.size_statistics->unencoded_byte_array_data_bytes)
                         : static_cast<std::size_t>(column_metadata.total_uncompressed_size);
          // Plus the cuDF string column's offsets (one int32 per row) and validity.
          rg_decoded += char_bytes + row_count * sizeof(std::uint32_t) + row_count / 8;
        }
      }
      rg_compressed += static_cast<std::size_t>(column_metadata.total_compressed_size);
    };
    if (_plan->is_projected()) {
      for (std::size_t i = 0; i < selected_chunk_indices.size(); ++i) {
        auto const chunk_idx = selected_chunk_indices[i];
        add_chunk(row_group.columns[chunk_idx],
                  pure_filter_chunk_indices.contains(chunk_idx),
                  selected_chunk_decoded_width[i]);
      }
    } else if (returned_types.size() == row_group.columns.size()) {
      // Unprojected (identity) scan: the reader materializes every file column
      // in order, so column ci aligns 1:1 with returned_types[ci]. Estimate
      // decoded bytes per column the same way as the projected path — fixed
      // widths from the type, VARCHAR/nested falling back to encoded size.
      for (std::size_t ci = 0; ci < row_group.columns.size(); ++ci) {
        std::size_t decoded_width = 0;
        try {
          decoded_width = returned_types[ci].fixed_width_byte_size();
        } catch (...) {
          decoded_width = 0;
        }
        add_chunk(row_group.columns[ci], /*is_pure_filter=*/false, decoded_width);
      }
    } else {
      // Column count does not match returned_types (cannot safely align types to
      // chunks): keep the original parquet encoded-uncompressed sizing.
      for (auto const& chunk : row_group.columns) {
        rg_decoded += static_cast<std::size_t>(chunk.meta_data.total_uncompressed_size);
        rg_compressed += static_cast<std::size_t>(chunk.meta_data.total_compressed_size);
      }
    }
    return std::pair{rg_decoded, rg_compressed};
  };

  // Cache-entry pre-sizing. Same decoded-byte estimate as rg_contribution above,
  // except pure-filter columns are INCLUDED: cache_entry_info::column_ids is built
  // from all of _plan->data_columns (trailing pure-filter columns included), so an
  // estimate that skipped them would under-count the entry the cache actually
  // builds. Deliberately a separate lambda rather than a flag on rg_contribution --
  // that value feeds split sizing, and perturbing it would change batch coalescing.
  auto rg_cache_contribution = [&](cudf::io::parquet::RowGroup const& row_group,
                                  std::vector<std::size_t>& per_column) {
    std::size_t rg_decoded = 0;
    auto const row_count   = static_cast<std::size_t>(row_group.num_rows);
    for (std::size_t i = 0; i < selected_chunk_indices.size(); ++i) {
      auto const& column_metadata = row_group.columns[selected_chunk_indices[i]].meta_data;
      auto const decoded_width    = selected_chunk_decoded_width[i];
      auto const before           = rg_decoded;
      if (decoded_width > 0) {
        rg_decoded += row_count * decoded_width + row_count / 8;
      } else {
        std::size_t const char_bytes =
          (column_metadata.size_statistics &&
           column_metadata.size_statistics->unencoded_byte_array_data_bytes)
            ? static_cast<std::size_t>(
                *column_metadata.size_statistics->unencoded_byte_array_data_bytes)
            : static_cast<std::size_t>(column_metadata.total_uncompressed_size);
        rg_decoded += char_bytes + row_count * sizeof(std::uint32_t) + row_count / 8;
      }
      auto const k = selected_chunk_column_idx[i];
      if (k < per_column.size()) { per_column[k] += rg_decoded - before; }
    }
    return rg_decoded;
  };

  auto out                     = std::make_unique<parquet_file_scan_info>();
  out->file_metadata           = file_metadata;
  out->file_path               = file_path;
  out->datasource              = std::move(sirius_ds);
  out->reader_options          = _reader_options;
  // What the split carries is the ROW-FILTER decision; materialize_table reads it
  // to decide whether to call set_filter on its own reader options.
  out->disable_filter_pushdown = !apply_row_filter;
  out->row_groups.reserve(row_group_indices.size());
  std::size_t projected_cache_bytes = 0;
  std::vector<std::size_t> per_column_bytes(data_column_names.size(), 0);
  for (auto const rg_idx : row_group_indices) {
    auto const& row_group        = metadata.row_groups[rg_idx];
    auto const [rg_unc, rg_comp] = rg_contribution(row_group);
    out->row_groups.push_back({rg_idx, rg_unc, rg_comp, row_group.num_rows});
    // Unprojected scans read every column, so rg_contribution's estimate already
    // covers the whole entry; only the projected path needs the pure-filter add-back.
    projected_cache_bytes +=
      _plan->is_projected() ? rg_cache_contribution(row_group, per_column_bytes) : rg_unc;
  }
  // Monotone across files: a multi-file scan can hand out an early split before the
  // total is known, so the gate below can admit a doomed entry until enough footers
  // have been read. reject_admission remains the backstop for that case.
  _projected_cache_entry_bytes.fetch_add(projected_cache_bytes, std::memory_order_relaxed);
  {
    std::lock_guard lock{_projected_column_bytes_mutex};
    auto& expected = _expected_row_groups[file_path];
    for (auto const rg_idx : row_group_indices) { expected.insert(static_cast<int>(rg_idx)); }
  }
  {
    std::lock_guard lock{_projected_column_bytes_mutex};
    if (_projected_column_bytes.size() < per_column_bytes.size()) {
      _projected_column_bytes.resize(per_column_bytes.size(), 0);
    }
    for (std::size_t k = 0; k < per_column_bytes.size(); ++k) {
      _projected_column_bytes[k] += per_column_bytes[k];
    }
  }

  // Hive partition values for this file, in scan_plan::partition_columns order.
  if (!_plan->partition_columns.empty()) {
    out->partition_values.reserve(_plan->partition_columns.size());
    auto parsed = duckdb::HivePartitioning::Parse(file_path);
    for (auto const& pc : _plan->partition_columns) {
      auto it = parsed.find(pc.name);
      out->partition_values.push_back(it != parsed.end() ? it->second : std::string{});
    }
  }

  return out;
}


std::unordered_map<std::string, std::unordered_set<int>>
parquet_gpu_ingestible::surviving_row_groups(io::ioctx_resolver const& resolve) const
{
  // Row groups this scan's STATIC predicate cannot rule out, from footer
  // statistics alone -- no data is read.
  //
  // Needed by the cache assignment, which happens before any split runs and so
  // cannot see build_file_scan_info's pruning. Without it a cached entry hands
  // back every row group it holds, including the ones the predicate would have
  // skipped; with pre-filter caching (pages hold unfiltered rows) that is the
  // whole point of the pruning thrown away.
  //
  // Static only: a dynamic filter has no value until the join's build side
  // publishes, which is after this runs. Those scans are refused by the cache
  // anyway.
  std::unordered_map<std::string, std::unordered_set<int>> out;
  if (!_duckdb_filter_expression || !resolve) { return out; }

  auto stream = cudf::get_default_stream();
  for (auto const& file_path : _file_paths) {
    std::shared_ptr<io::sirius_ioctx> io_ctx;
    try {
      io_ctx = resolve(file_path);
    } catch (...) {
      return {};  // cannot resolve one file: claim nothing rather than a wrong subset
    }
    if (!io_ctx) { return {}; }
    auto sirius_ds = io_ctx->open_datasource(file_path);
    if (!sirius_ds) { return {}; }

    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    if (auto cached = sirius_ds->metadata()) {
      if (auto pm = std::dynamic_pointer_cast<parquet_metadata>(std::move(cached))) {
        file_metadata = pm->file_metadata();
      }
    }
    if (!file_metadata) { return {}; }  // no parked footer: not worth a fetch here

    auto opts = *_reader_options;
    std::optional<gpu_expression_translator::translated_expression> ast;
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    try {
      gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
      auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
      ast = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
      if (!ast) { return {}; }
      opts.set_filter(ast->back());
      hybrid_scan_reader reader(*file_metadata, opts);
      auto const survivors = reader.filter_row_groups_with_stats(
        reader.all_row_groups(opts), opts, stream);
      auto& set = out[file_path];
      for (auto const rg : survivors) { set.insert(static_cast<int>(rg)); }
    } catch (...) {
      return {};  // translation or pruning failed: claim nothing
    }
  }
  return out;
}

std::string parquet_gpu_ingestible::fixed_page_cache_filter_signature() const
{
  return _duckdb_filter_expression ? _duckdb_filter_expression->ToString() : std::string{};
}

namespace {

/// Fold one comparison into a value range, or fail.
///
/// Only the shapes a containment test can reason about: a bare column reference
/// compared against a constant. Anything else -- a function call, a cast, two
/// columns, LIKE, <> -- leaves the predicate unanalyzable, and the caller then
/// falls back to matching the predicate's text exactly, which is what it did
/// before this existed.
bool fold_comparison(duckdb::BoundComparisonExpression const& expr,
                     std::function<std::string(duckdb::idx_t)> const& column_of,
                     std::vector<scan_manager::cache_filter_range>& out)
{
  auto const* ref   = dynamic_cast<duckdb::BoundReferenceExpression const*>(expr.left.get());
  auto const* konst = dynamic_cast<duckdb::BoundConstantExpression const*>(expr.right.get());
  auto type         = expr.GetExpressionType();
  if (ref == nullptr || konst == nullptr) {
    // Try the mirrored form (constant on the left) and flip the operator with it.
    ref   = dynamic_cast<duckdb::BoundReferenceExpression const*>(expr.right.get());
    konst = dynamic_cast<duckdb::BoundConstantExpression const*>(expr.left.get());
    if (ref == nullptr || konst == nullptr) { return false; }
    switch (type) {
      case duckdb::ExpressionType::COMPARE_LESSTHAN:
        type = duckdb::ExpressionType::COMPARE_GREATERTHAN; break;
      case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
        type = duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHAN:
        type = duckdb::ExpressionType::COMPARE_LESSTHAN; break;
      case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        type = duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO; break;
      default: break;
    }
  }
  if (konst->value.IsNull()) { return false; }

  scan_manager::cache_filter_range range;
  range.column_name = column_of(ref->index);
  if (range.column_name.empty()) { return false; }
  switch (type) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
      range.has_lo = range.has_hi = true;
      range.lo = range.hi = konst->value;
      break;
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
      range.has_lo = true; range.lo = konst->value; range.lo_inclusive = false; break;
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      range.has_lo = true; range.lo = konst->value; break;
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
      range.has_hi = true; range.hi = konst->value; range.hi_inclusive = false; break;
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      range.has_hi = true; range.hi = konst->value; break;
    default: return false;
  }
  out.push_back(std::move(range));
  return true;
}

/// Walk a conjunction of comparisons, collecting the ones that are value ranges.
///
/// @p strict decides what an unreadable conjunct means, and the two sides of the
/// containment test need opposite answers:
///
///  - PRODUCER (strict): the cache holds {r : A and B}. Not knowing B means not
///    knowing which rows were dropped, so a consumer cannot be cleared against it.
///    Fail the whole predicate.
///  - CONSUMER (lenient): the query wants {r : C and D}, which is a SUBSET of
///    {r : C}. If C alone falls inside the producer's range then so does C and D,
///    so an unreadable D can simply be dropped -- it only narrows the request.
///
/// Treating both sides strictly is what made this fire zero times on ClickBench:
/// q37-q43 each carry one `<> ''` or `contains(...)`, enough to throw away the
/// CounterID and EventDate ranges sitting next to it.
bool collect_ranges(duckdb::Expression const& expr,
                    std::function<std::string(duckdb::idx_t)> const& column_of,
                    std::vector<scan_manager::cache_filter_range>& out,
                    bool strict)
{
  if (expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_AND) {
    auto const& conj = expr.Cast<duckdb::BoundConjunctionExpression>();
    for (auto const& child : conj.children) {
      if (!collect_ranges(*child, column_of, out, strict) && strict) { return false; }
    }
    return true;
  }
  if (auto const* cmp = dynamic_cast<duckdb::BoundComparisonExpression const*>(&expr)) {
    return fold_comparison(*cmp, column_of, out);
  }
  return false;
}

/// Flatten a predicate into its AND-ed parts, each as its own text.
///
/// The range test can only compare predicates it can turn into min/max bounds,
/// and ClickBench's are mostly `<>` and LIKE, which have no bounds: 292 of 328
/// recorded predicates came back unanalyzable, and the lookup then fell through
/// to comparing the whole predicate string, which matches only an identical
/// query. 88 of 110 cache lookups were refused that way.
///
/// Conjuncts need no bounds. A page filtered by `A` serves a query filtered by
/// `A AND B` for any B, because B only removes rows -- so the producer's parts
/// being a SUBSET of the consumer's is enough, whatever the parts say.
void collect_conjuncts(duckdb::Expression const& expr, std::vector<std::string>& out)
{
  if (expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_AND) {
    auto const& conj = expr.Cast<duckdb::BoundConjunctionExpression>();
    for (auto const& child : conj.children) { collect_conjuncts(*child, out); }
    return;
  }
  out.push_back(expr.ToString());
}

}  // namespace

std::vector<std::string> parquet_gpu_ingestible::fixed_page_cache_filter_conjuncts() const
{
  std::vector<std::string> out;
  if (_duckdb_filter_expression) { collect_conjuncts(*_duckdb_filter_expression, out); }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<scan_manager::cache_filter_range> parquet_gpu_ingestible::fixed_page_cache_filter_ranges(
  bool& analyzable, bool strict) const
{
  std::vector<scan_manager::cache_filter_range> ranges;
  if (!_duckdb_filter_expression) {
    analyzable = true;  // no predicate: selects everything, subsumes any query
    return ranges;
  }
  auto column_of = [this](duckdb::idx_t ref_index) { return _plan->batch_column_name(ref_index); };
  analyzable     = collect_ranges(*_duckdb_filter_expression, column_of, ranges, strict);
  if (!analyzable) { ranges.clear(); }
  SIRIUS_LOG_INFO("[fixed-page-cache] filter_ranges strict={} analyzable={} count={} root='{}' expr='{}'",
                  strict,
                  analyzable,
                  ranges.size(),
                  duckdb::ExpressionTypeToString(_duckdb_filter_expression->GetExpressionType()),
                  _duckdb_filter_expression->ToString().substr(0, 120));
  return ranges;
}


bool parquet_gpu_ingestible::fixed_page_cache_has_dynamic_filters() const
{
  return static_cast<bool>(_sirius_dynamic_filters);
}

/// Whether a scan carrying dynamic filters may still populate the page cache.
///
/// Such a scan is skipped by default, which on TPC-H/JCC-H means most of lineitem: a
/// probe-side scan almost always has a join's dynamic filter attached (measured on an
/// SF50 22-query run: 675 of 795 `dynamic_filter_scan` skips were lineitem, 58 orders).
/// Nothing about the decoded data forces that, though -- `disable_filter_pushdown`
/// already keeps dynamic filters off the reader whenever auto-caching is on, so the
/// reader hands back the whole column exactly as it does for a static predicate, and
/// cached batches leave as filter_state::UNFILTERED for each consumer to re-filter.
/// The real trade is that a dynamic filter can be far more selective than a static one,
/// so caching the unfiltered column costs I/O and cache capacity for rows the query
/// would have skipped. Off by default so that trade stays opt-in and measurable.
bool dynamic_filter_scan_cache_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_DYNAMIC_FILTER_SCANS");
  return value != nullptr && std::string_view(value) == "1";
}

/// Whether to reject an oversized cache entry from the parquet footer, before the
/// decoded copy is made.
///
/// The cache already rejects entries larger than its per-entry ceiling, but only
/// after each chunk has been decoded and copied into the entry -- so a table that
/// can never fit still inflates resident bytes on its way to being erased. That
/// transient pushes the cache over its budget and fires the per-page LRU, which
/// evicts pages belonging to OTHER, well-sized entries; those entries then fail
/// `incomplete resident coverage` on every later query and nothing repopulates
/// them. Measured on an SF50 22-query run: 58.9 GiB copied-then-erased, peak
/// resident 5.61 GiB against a 5.5 GiB target, 36 page-budget evictions, and 87
/// coverage failures that persisted through executions 2 and 3.
///
/// Off by default: the footer estimate can under-count (a writer that omits
/// SizeStatistics leaves dictionary-encoded strings sized by their ENCODED bytes),
/// and an over-count costs a cache hit that would have been legal.
/// How large a table may be, as a multiple of the page-cache budget, and still be
/// worth caching from a scan that carries a join's dynamic filter.
///
/// Such a scan is normally refused, because caching it means caching the rows the
/// join would have discarded: auto-caching forces the reader's filter off, so the
/// whole column comes back. The right fix would be to keep the reader's ROW-GROUP
/// pruning while dropping only the row masking -- the two are already separate
/// flags here -- but the dynamic filter is built from the join's build side and is
/// still empty when pruning runs (measured on SSB SF50: 26 of 26 lineorder scans
/// saw has_filters=0), so pruning has nothing to prune with.
///
/// What is left is to ask whether reading the table whole is affordable at all.
/// Measured with the write gate forced on, the answer flips with the table's size
/// against the budget: SSB SF30 (working set 1.44x the budget) gained 11.1%, SF50
/// (2.40x) lost 10.8%. Default 1.5 sits at that boundary; 0 restores the old
/// unconditional refusal.
double dynamic_filter_cache_size_ratio()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_DYNAMIC_FILTER_SIZE_RATIO");
  if (value == nullptr) { return 1.5; }
  try {
    return std::max(0.0, std::stod(value));
  } catch (std::exception const&) {
    SIRIUS_LOG_WARN("[fixed-page-cache] invalid SIRIUS_FIXED_PAGE_DYNAMIC_FILTER_SIZE_RATIO='{}'",
                    value);
    return 1.5;
  }
}

bool presize_admission_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_PRESIZE_ADMISSION");
  return value != nullptr && std::string_view(value) == "1";
}


void parquet_gpu_ingestible::set_cached_row_groups(
  std::unordered_map<std::string, std::unordered_set<int>> groups)
{
  _cached_row_groups = std::move(groups);
}

void parquet_gpu_ingestible::auto_cache_materialized_table(
  cudf::table_view view,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  bool reader_applied_filter,
  bool reader_applied_dynamic_filter,
  std::vector<row_group_slice> const& rg_slices)
{
  if (!_scan_manager || !fixed_page_auto_cache_enabled() || view.num_columns() == 0 ||
      view.num_rows() == 0 || _plan->has_partitions()) {
    return;
  }
  if (fixed_page_cache_has_dynamic_filters() && !dynamic_filter_scan_cache_enabled()) {
    // Not an unconditional refusal any more: a table small enough against the
    // budget is worth caching whole, because what lands in the cache carries no
    // predicate and every later query can read it. A table that is not small
    // enough is refused exactly as before -- reading it whole costs more than the
    // hits it buys.
    auto const projected = _projected_cache_entry_bytes.load(std::memory_order_relaxed);
    auto const budget    = scan_manager::sirius_scan_manager::fixed_page_cache_budget_bytes();
    auto const ratio     = dynamic_filter_cache_size_ratio();
    auto const ceiling   = static_cast<std::size_t>(static_cast<double>(budget) * ratio);
    bool const small_enough =
      ratio > 0.0 && projected != 0 && budget != 0 && projected <= ceiling;
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] dynamic_filter_size_gate projected_bytes={} budget_bytes={} "
      "ratio={} ceiling_bytes={} cached={}",
      projected,
      budget,
      ratio,
      ceiling,
      small_enough ? 1 : 0);
    if (!small_enough) {
      SIRIUS_LOG_INFO("[fixed-page-cache] auto_cache_skip reason=dynamic_filter_scan");
      return;
    }
  }

  bool has_fixed_width_column = false;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    bool const fixed_width = is_fixed_width_auto_cache_candidate(view.column(i));
    has_fixed_width_column = has_fixed_width_column || fixed_width;
    if (!fixed_width && !fixed_page_hybrid_provider_enabled()) {
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] auto_cache_skip reason=non_fixed_width_column column_index={} type_id={}",
        i,
        static_cast<int>(view.column(i).type().id()));
      return;
    }
  }
  // NOTE: a batch of nothing but STRING columns is refused here even when the
  // variable-width index is on, which makes insert_fixed_page_entry_from_view's
  // widening for that case ("so a batch made up entirely of STRING columns ...
  // still reaches the per-column loop") unreachable. That looks like a bug and
  // it was measured as one -- 120 skips on ClickBench, all of them `URL` (90)
  // and `URL|SearchPhrase` (30), the projections of the four queries holding
  // 17.1s of scan time the fixed-width cache cannot touch.
  //
  // Opening it was tried and reverted: with no per-column size admission, a
  // URL-only entry (~9 GB decoded against a 2 GB variable budget) is admitted
  // and then thrashes, and q35 OOMs at the 22 GB device limit with the cache
  // holding only 1.6 GB. The gate can only be opened together with a per-column
  // decision that refuses a column too large to ever be resident -- at which
  // point a URL-only batch is refused again anyway, just more cheaply.
  //
  // That reasoning is about the pinned-entry store. The page store pages STRING columns
  // in the same store, budget and LRU as everything else, so the gate now keeps string-only
  // batches out of the pinned store only; they still reach the page store below. Measured on
  // ClickBench before this: 147 scans (URL, URL|SearchPhrase, SearchPhrase) were never
  // cached at all. SIRIUS_PAGE_STORE_STRING_ONLY=0 restores the old refusal.
  bool const string_only_batch = !has_fixed_width_column;
  if (string_only_batch && !page_store_string_only_enabled()) {
    SIRIUS_LOG_INFO("[fixed-page-cache] auto_cache_skip reason=no_fixed_width_column");
    return;
  }
  if (reader_applied_filter && !cache_filtered_scans_enabled()) {
    SIRIUS_LOG_INFO("[fixed-page-cache] auto_cache_skip reason=filtered_scan");
    return;
  }

  scan_manager::cache_entry_info cache_info;
  cache_info.resolved_file_paths = _file_paths;
  // Only claim a filter identity when the reader actually applied one. When AST
  // translation rejects the predicate (LIKE / contains / prefix / suffix /
  // substring) the reader returns the FULL table, and this function runs before
  // post_filter_and_project -- so the view being cached is unfiltered. Stamping a
  // filter signature on it made an unfiltered table masquerade as a filtered one:
  // it could only ever be reused by a query carrying the byte-identical predicate,
  // even though it is valid for any query over the same columns. Measured on SF30:
  // 7 of 22 filter-stamped entries held 100% of their base table.
  //
  // Leaving the signature empty is safe because cached batches are handed out as
  // filter_state::UNFILTERED (gpu_ingestible.cpp), so post_filter_and_project
  // re-applies each consuming query's own predicate post-decode.
  cache_info.filter_signature =
    (reader_applied_filter || !honest_filter_signature_enabled())
      ? fixed_page_cache_filter_signature()
      : std::string{};
  // Record the same predicate as ranges so a later scan can ask whether its own
  // predicate is narrower, instead of only whether the text matches byte for byte.
  if (!cache_info.filter_signature.empty()) {
    bool analyzable          = false;
    cache_info.filter_ranges = fixed_page_cache_filter_ranges(analyzable, /*strict=*/true);
    cache_info.filter_analyzable = analyzable;
  } else {
    cache_info.filter_analyzable = true;  // unfiltered: selects everything
  }
  cache_info.chunks_carry_filter_ranges = filter_keyless_cache_enabled();
  cache_info.column_ids.reserve(_plan->data_columns.size());
  cache_info.names.reserve(_plan->data_columns.size());
  for (auto const& dc : _plan->data_columns) {
    cache_info.column_ids.emplace_back(duckdb::ColumnIndex(dc.primary_idx));
    cache_info.names.push_back(dc.name);
  }
  {
    std::lock_guard lock{_projected_column_bytes_mutex};
    if (_projected_column_bytes.size() == cache_info.names.size()) {
      cache_info.projected_column_bytes = _projected_column_bytes;
    }
  }
  // Row groups this batch came from, and the table's true row count. Together they
  // are what lets a later query tell a complete entry from a short one, and name
  // the rows a partial entry does NOT hold.
  scan_manager::chunk_provenance provenance;
  provenance.num_rows = static_cast<std::size_t>(view.num_rows());
  // The predicate these rows came through, so a later scan can tell whether its own
  // is narrower. Strict: an unreadable conjunct means the rows that were dropped
  // cannot be characterised, and the chunk then serves only an identical predicate.
  if (reader_applied_filter) {
    bool analyzable             = false;
    provenance.filter_ranges    = fixed_page_cache_filter_ranges(analyzable, /*strict=*/true);
    provenance.filter_analyzable = analyzable;
    provenance.filter_signature  = fixed_page_cache_filter_signature();
  } else {
    provenance.filter_analyzable = true;  // unfiltered rows subsume any request
  }
  provenance.slices.reserve(rg_slices.size());
  for (auto const& slice : rg_slices) {
    provenance.slices.emplace_back(slice.file_path, slice.row_group_indices);
    // Per-row-group row counts, flattened in slice order, so the page cutter can
    // stop a page at every row-group boundary.
    if (slice.file_metadata) {
      for (auto const rg : slice.row_group_indices) {
        auto const idx = static_cast<std::size_t>(rg);
        if (idx < slice.file_metadata->row_groups.size()) {
          provenance.row_group_rows.push_back(
            static_cast<std::size_t>(slice.file_metadata->row_groups[idx].num_rows));
        }
      }
    }
  }
  // Flat page store: one page per (file, column, row group). This is what a later
  // scan looks the cache up in -- no entry, no projection, no predicate in the key.
  //
  // Not when the reader applied a dynamic filter. Those rows passed the join's runtime
  // filter as well as the static predicate, but a page's filter identity is the static
  // predicate's text alone -- and with no static predicate it is empty, which reads as
  // "every row of the row group". The splice path survives such a page only because its
  // row-count check rejects the mismatch; a full-hit read has no reader output to check
  // against and would silently drop the rows the runtime filter removed. Measured on TPC-H
  // SF50: 525 scans carrying dynamic filters were cached through the size gate.
  bool const page_store_takes_batch = _scan_manager != nullptr && rg_slices.size() == 1;
  if (page_store_takes_batch && reader_applied_dynamic_filter) {
    SIRIUS_LOG_INFO("[page-store] insert_skip reason=reader_applied_dynamic_filter file='{}'",
                    rg_slices.front().file_path);
  }
  if (page_store_takes_batch && !reader_applied_dynamic_filter) {
    auto const t_pages0 = std::chrono::steady_clock::now();
    // How many row groups the file has, so the store can tell a column that could be fully
    // resident from one that never can. Zero when the metadata is not to hand, which the
    // store reads as "unknown" and admits as before. Row GROUPS, not rows: the store holds
    // post-filter rows, so scaling this row group's rows up by the file's row count
    // over-counts wildly on a selective scan -- it read a 16 MiB AdvEngineID as 12.77 GiB.
    std::size_t file_row_groups = 0;
    if (auto const& md = rg_slices.front().file_metadata; md) {
      file_row_groups = md->row_groups.size();
    }
    _scan_manager->insert_pages_from_view(rg_slices.front().file_path,
                                          rg_slices.front().row_group_indices,
                                          provenance.row_group_rows,
                                          cache_info.names,
                                          view,
                                          provenance.filter_signature,
                                          reader_applied_filter
                                            ? fixed_page_cache_filter_conjuncts()
                                            : std::vector<std::string>{},
                                          provenance.filter_ranges,
                                          provenance.filter_analyzable,
                                          const_cast<cucascade::memory::memory_space&>(mem_space),
                                          stream,
                                          file_row_groups);
    auto const [bytes, pages] = _scan_manager->page_store_size();
    SIRIUS_LOG_INFO("[page-store] insert file='{}' row_groups={} columns={} -> pages={} bytes={}",
                    rg_slices.front().file_path,
                    rg_slices.front().row_group_indices.size(),
                    cache_info.names.size(),
                    pages,
                    bytes);
    SIRIUS_LOG_INFO("[scan-populate] store=pages us={}",
                    std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t_pages0)
                      .count());
  }
  if (page_store_takes_batch && !pinned_entry_cache_for_paged_batches_enabled()) { return; }
  if (string_only_batch) {
    SIRIUS_LOG_INFO("[fixed-page-cache] auto_cache_skip reason=no_fixed_width_column store=pinned");
    return;
  }
  auto const t_pinned0 = std::chrono::steady_clock::now();
  struct pinned_timer {
    std::chrono::steady_clock::time_point t0;
    ~pinned_timer()
    {
      SIRIUS_LOG_INFO("[scan-populate] store=pinned us={}",
                      std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count());
    }
  } const pinned_timer_guard{t_pinned0};

  std::size_t table_total_rows = 0;
  for (auto const& slice : rg_slices) {
    if (!slice.file_metadata) {
      table_total_rows = 0;
      break;
    }
    table_total_rows += static_cast<std::size_t>(slice.file_metadata->num_rows);
  }
  // A multi-file scan sums each file once; a slice per file repeats it, so only
  // trust the total when every file appears exactly once.
  {
    std::unordered_set<std::string> seen;
    std::size_t recomputed = 0;
    bool ok                = true;
    for (auto const& slice : rg_slices) {
      if (!slice.file_metadata) { ok = false; break; }
      if (seen.insert(slice.file_path).second) {
        recomputed += static_cast<std::size_t>(slice.file_metadata->num_rows);
      }
    }
    table_total_rows = ok ? recomputed : 0;
  }
  cache_info.table_total_rows = table_total_rows;
  {
    std::lock_guard lock{_projected_column_bytes_mutex};
    cache_info.expected_row_groups.assign(_expected_row_groups.begin(),
                                          _expected_row_groups.end());
  }
  if (cache_info.column_ids.size() != static_cast<std::size_t>(view.num_columns())) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] auto_cache_skip reason=column_count_mismatch expected={} actual={}",
      cache_info.column_ids.size(),
      view.num_columns());
    return;
  }

  auto const name =
    auto_fixed_page_cache_name(_file_paths, cache_info.filter_signature, cache_info.names);

  // Footer-based fixed-width pre-rejection. This is deliberately NOT an early
  // return: the verdict is about the fixed-width columns only, and vetoing the
  // whole insert also disables the variable-width (STRING) page index, which has
  // its own separate budget. Measured on SF50 when this did return early:
  // lineitem's variable-width indexing dropped 61 -> 0 and scan time rose 2.10%
  // -> 4.18% over the fixed-only baseline, wiping out the gain from removing
  // 53.6 GiB of copied-then-erased admission waste.
  bool fixed_width_pre_rejected = false;
  if (presize_admission_enabled()) {
    auto const projected_bytes = _projected_cache_entry_bytes.load(std::memory_order_relaxed);
    auto const admission_limit =
      scan_manager::sirius_scan_manager::fixed_page_admission_limit_bytes();
    fixed_width_pre_rejected =
      projected_bytes != 0 && admission_limit != 0 && projected_bytes > admission_limit;
    // projected_bytes is logged on both paths: comparing it against the eventual
    // incoming_bytes on the admission log line is how the footer estimate's
    // accuracy gets checked (notably whether cuDF widens FLBA decimals to
    // decimal128, which would make this estimate 2x low for the decimal columns).
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] projected_entry_bytes table='{}' projected_bytes={} "
      "max_entry_bytes={} fixed_width_pre_rejected={}",
      name,
      projected_bytes,
      admission_limit,
      fixed_width_pre_rejected);
  }

  try {
    if (fixed_page_owned_pages_enabled() && fixed_page_direct_auto_populate_enabled()) {
      auto const pages_added = _scan_manager->insert_fixed_page_entry_from_view(
        name,
        std::move(cache_info),
        view,
        const_cast<cucascade::memory::memory_space&>(mem_space),
        stream,
        fixed_width_pre_rejected,
        provenance);
      if (pages_added) {
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] auto_cache_populate_direct table='{}' rows={} columns={} pages={}",
          name,
          view.num_rows(),
          view.num_columns(),
          pages_added);
      }
      return;
    }

    auto cache_table = std::make_unique<cudf::table>(view, stream, mem_space.get_default_allocator());
    std::vector<std::unique_ptr<cudf::table>> tables;
    tables.push_back(std::move(cache_table));
    std::vector<cucascade::memory::memory_space*> spaces;
    spaces.push_back(const_cast<cucascade::memory::memory_space*>(&mem_space));

    _scan_manager->insert_pinned_entry(name, std::move(cache_info), std::move(tables), spaces);
    SIRIUS_LOG_INFO("[fixed-page-cache] auto_cache_populate table='{}' rows={} columns={}",
                    name,
                    view.num_rows(),
                    view.num_columns());
  } catch (std::exception const& ex) {
    SIRIUS_LOG_WARN("[fixed-page-cache] auto_cache_populate_failed error='{}'", ex.what());
  }
}

//===----------------------------------------------------------------------===//
// materialize_table — ports read_table_from_metadata
//===----------------------------------------------------------------------===//
filtered_table parquet_gpu_ingestible::materialize_metadata_to_table(
  op::scan::scan_info const& info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& split = static_cast<parquet_split_info const&>(info);

  // LATE_MATERIALIZATION's re-fetch scan: the join that consumes it publishes the build side's
  // file_row_number range. Drop the row groups outside it before reading anything -- that range
  // is what turns "decode every column of every row" into "decode the few row groups the TOP_N
  // picked". A pruned copy goes back through this function; a second pass prunes nothing.
  if (auto const rn_pos = _plan->file_row_number_output_position();
      rn_pos && _sirius_dynamic_filters && _sirius_dynamic_filters->has_filters()) {
    // Exact path: the join's IN-list holds the few row numbers wanted (TOP_N's N). Read just
    // those rows -- a skip_rows/num_rows read per run of nearby rows -- not their row groups,
    // which for q24 are spread over the whole file.
    std::vector<std::int64_t> const* exact = nullptr;
    for (auto const& f : _sirius_dynamic_filters->filters_for_column(*rn_pos)) {
      if (auto const* in_list = dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(f.get())) {
        if (auto const* keys = in_list->host_keys()) { exact = keys; }
      }
    }
    if (exact && !_plan->has_partitions()) {
      return read_row_numbers(split, *exact, mem_space, stream);
    }
    std::optional<std::int64_t> lo, hi;
    for (auto const& f : _sirius_dynamic_filters->filters_for_column(*rn_pos)) {
      // Joins publish the exact IN-list by default (zone maps are off); its recorded key range
      // is all the pruning needs.
      if (auto const* in_list = dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(f.get())) {
        if (auto const b = in_list->key_bounds()) {
          lo = lo ? std::min(*lo, b->first) : b->first;
          hi = hi ? std::max(*hi, b->second) : b->second;
        }
        continue;
      }
      auto const* zm = dynamic_cast<sirius::op::sirius_dynamic_zone_map_filter const*>(f.get());
      if (!zm) { continue; }
      for (auto const& z : zm->zones()) {
        if (!z.min || !z.max || z.min->type().id() != cudf::type_id::INT64) { continue; }
        auto const zmin = static_cast<cudf::numeric_scalar<std::int64_t> const&>(*z.min).value(stream);
        auto const zmax = static_cast<cudf::numeric_scalar<std::int64_t> const&>(*z.max).value(stream);
        lo = lo ? std::min(*lo, zmin) : zmin;
        hi = hi ? std::max(*hi, zmax) : zmax;
      }
    }
    if (lo && hi) {
      parquet_split_info pruned = split;
      std::size_t before = 0, after = 0;
      for (auto& slice : pruned.rg_slices) {
        if (!slice.file_metadata) { continue; }
        auto const& rgs = slice.file_metadata->row_groups;
        std::vector<std::int64_t> start(rgs.size() + 1, 0);
        for (std::size_t i = 0; i < rgs.size(); ++i) {
          start[i + 1] = start[i] + static_cast<std::int64_t>(rgs[i].num_rows);
        }
        std::vector<cudf::size_type> kept;
        for (auto const rg : slice.row_group_indices) {
          auto const r = static_cast<std::size_t>(rg);
          if (start[r + 1] - 1 >= *lo && start[r] <= *hi) { kept.push_back(rg); }
        }
        before += slice.row_group_indices.size();
        after += kept.size();
        slice.row_group_indices = std::move(kept);
      }
      if (after < before) {
        SIRIUS_LOG_INFO("[late-materialization] file_row_number in [{}, {}]: {} of {} row groups",
                        *lo, *hi, after, before);
        return materialize_metadata_to_table(pruned, mem_space, stream);
      }
    }
  }

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
      sources.push_back(cudf::io::datasource::create(slice.file_path));
    }
    metadatas.push_back(*slice.file_metadata);
    rg_per_src.push_back(slice.row_group_indices);
  }
  // All-pruned fallback split (parquet_batch_coalescer::flush): every slice
  // carries zero row groups.
  // Don't express that via set_row_groups — the meaning of an empty per-source
  // vector has flipped between cudf versions ("all row groups" vs "none").
  // Instead bound the read to zero rows against the footer metadata alone:
  // cudf builds the schema-correct empty table without touching data pages,
  // and it flows through the normal filter / partition / projection assembly
  // below.
  bool const all_slices_pruned =
    !split.rg_slices.empty() &&
    std::all_of(split.rg_slices.begin(), split.rg_slices.end(), [](row_group_slice const& s) {
      return s.row_group_indices.empty();
    });
  auto opts = *split.reader_options;
  if (all_slices_pruned) {
    opts.set_num_rows(0);
  } else {
    opts.set_row_groups(std::move(rg_per_src));
  }

  // Per-task AST translation for reader-side row-group + row pushdown. set_filter
  // is gated on translation success AND on the per-batch disable_filter_pushdown
  // flag (set when the FLBA-decimal probe failed). When pushdown does not engage
  // — disabled, translation fails, or the split is the all-pruned zero-row
  // fallback (zero rows need no reader filter; skipping keeps GPU AST
  // translation off that path) — the row filter is left for
  // post_filter_and_project to apply post-decode. The translated cuDF AST
  // (`ast_expression`) must outlive read_parquet; the borrowed Sirius AST and
  // the translator are only needed during translation.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  std::optional<gpu_expression_translator::translated_expression> dynamic_ast_expression =
    std::nullopt;
  // The audit line opens at read_parquet, so filter translation and split preparation
  // before it were never measured.
  auto const t_stage_begin = std::chrono::steady_clock::now();
  std::int64_t stage_prep_us = 0;
  cudf::ast::expression const* reader_filter_root = nullptr;

  bool const auto_cache = fixed_page_auto_cache_enabled() && _scan_manager != nullptr;

  if (_duckdb_filter_expression && !split.disable_filter_pushdown && !all_slices_pruned) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    auto name_resolver     = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
      return plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { reader_filter_root = &ast_expression->back(); }
  }

  // Whether the reader's filter includes a dynamic (join runtime) filter. Those rows are
  // not described by the static predicate text the cache keys pages on.
  bool reader_applied_dynamic_filter = false;
  if (!split.disable_filter_pushdown && _sirius_dynamic_filters &&
      _sirius_dynamic_filters->has_filters()) {
    auto const* const root_before_dynamic = reader_filter_root;
    if (ast_expression) {
      reader_filter_root = merge_dynamic_filters_into_ast(ast_expression->tree,
                                                          reader_filter_root,
                                                          *_sirius_dynamic_filters,
                                                          *split.plan,
                                                          mem_space.get_device_id());
    } else {
      dynamic_ast_expression.emplace();
      reader_filter_root = merge_dynamic_filters_into_ast(dynamic_ast_expression->tree,
                                                          /*existing_root=*/nullptr,
                                                          *_sirius_dynamic_filters,
                                                          *split.plan,
                                                          mem_space.get_device_id());
      if (!reader_filter_root) { dynamic_ast_expression.reset(); }
    }
    reader_applied_dynamic_filter =
      reader_filter_root != nullptr && reader_filter_root != root_before_dynamic;
  }

  if (reader_filter_root) { opts.set_filter(*reader_filter_root); }
  auto const filter_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t_stage_begin).count();

  std::vector<std::string> audit_files;
  audit_files.reserve(split.rg_slices.size());
  std::size_t audit_uncompressed_bytes = 0;
  std::size_t audit_compressed_bytes   = 0;
  for (auto const& slice : split.rg_slices) {
    audit_files.push_back(slice.file_path);
    audit_uncompressed_bytes += slice.reserved_uncompressed_bytes;
    audit_compressed_bytes += slice.reserved_compressed_bytes;
  }
  auto const audit_columns    = join_strings(split.plan->data_column_names(), '|');
  auto const audit_row_groups = scan_audit_row_groups(split.rg_slices);

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());

  // Columns already on the GPU for exactly these row groups. Read only the ones
  // that are missing and paste the cached ones back in below, instead of sending
  // every column to parquet because one of them was absent.
  //
  // Sound only when this read is UNFILTERED and covers whole row groups: the
  // cached data holds all of the row group's rows in file order, so it lines up
  // with freshly read columns row for row. A reader-side filter would drop rows
  // from the new columns but not from the cached ones.
  // Where a scan's time goes, so the cache's share of it stops being a guess.
  std::int64_t scan_probe_lookup_us = 0, scan_probe_materialize_us = 0;
  std::int64_t scan_probe_read_us = 0, scan_probe_assemble_us = 0;
  // Partial serving's share of stage_prep_us: page lookup, disk reads of the missing ranges,
  // and concatenating cached and read pieces.
  std::int64_t ps_peek_us = 0, ps_read_us = 0, ps_assemble_us = 0;
  auto elapsed_us = [](auto since) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - since).count();
  };
  std::vector<std::shared_ptr<cudf::column>> spliced;
  std::vector<std::string> read_names;
  auto const projected_names = split.plan->data_column_names();
  bool splicing              = false;
  bool full_hit              = false;
  std::size_t trace_available = 0;  // columns the probe found usable, for the scan-cache trace

  // Partial serving (HANDOFF B4). The splice below needs a column cached for every row of
  // every row group in the split; one page missing anywhere sends the whole column to the
  // reader. Here a column takes the pages it has and reads only the row ranges it lacks.
  // Limited to an unfiltered read of one file, where cached rows and read rows line up by
  // position; everything else keeps the paths below.
  bool partial_hit = false;
  std::size_t hit_read_bytes = 0;  // SIRIUS_CACHE_HIT_TRACE: decoded bytes the reader produced
  std::unique_ptr<cudf::table> partial_table;
  if (_scan_manager != nullptr && !all_slices_pruned && split.rg_slices.size() == 1 &&
      !projected_names.empty() && reader_filter_root == nullptr && !_plan->has_partitions() &&
      page_partial_serve_enabled() && split.rg_slices.front().file_metadata) {
    auto const& slice = split.rg_slices.front();
    std::vector<std::string> const wanted(projected_names.begin(), projected_names.end());
    auto const& rgs      = slice.row_group_indices;
    auto const& file_rgs = slice.file_metadata->row_groups;
    // File-global first row and row count of each requested row group.
    std::vector<std::int64_t> prefix(file_rgs.size() + 1, 0);
    for (std::size_t i = 0; i < file_rgs.size(); ++i) {
      prefix[i + 1] = prefix[i] + static_cast<std::int64_t>(file_rgs[i].num_rows);
    }
    std::vector<std::int64_t> rg_start(rgs.size()), rg_len(rgs.size());
    for (std::size_t g = 0; g < rgs.size(); ++g) {
      rg_start[g] = prefix[static_cast<std::size_t>(rgs[g])];
      rg_len[g]   = static_cast<std::int64_t>(file_rgs[static_cast<std::size_t>(rgs[g])].num_rows);
    }
    auto const total_rows = std::accumulate(rg_len.begin(), rg_len.end(), std::int64_t{0});
    auto covered = [](std::vector<std::vector<scan_manager::cached_piece>> const& per_rg) {
      std::int64_t c = 0;
      for (auto const& v : per_rg) {
        for (auto const& p : v) { c += static_cast<std::int64_t>(p.num_rows); }
      }
      return c;
    };
    // Peek first: the path is only worth taking when some column is part-cached. Fully cached
    // and fully missing columns are what the full-hit and splice paths below already handle.
    auto const t_peek0 = std::chrono::steady_clock::now();
    auto const peek =
      _scan_manager->cached_row_group_pieces(slice.file_path, rgs, wanted, std::string{}, false);
    bool any_partial = false;
    for (auto const& per_rg : peek) {
      auto const c = covered(per_rg);
      if (c > 0 && c < total_rows) {
        any_partial = true;
        break;
      }
    }
    ps_peek_us = elapsed_us(t_peek0);
    if (any_partial) {
      auto const t_ps0 = std::chrono::steady_clock::now();
      auto pieces =
        _scan_manager->cached_row_group_pieces(slice.file_path, rgs, wanted, std::string{}, true);
      struct segment {
        std::int64_t start{0};
        std::int64_t len{0};
        bool cached{false};
        cudf::column_view view{};
      };
      std::vector<std::vector<segment>> segs(wanted.size());
      std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::size_t>> missing;
      bool layout_ok = true;
      for (std::size_t ci = 0; ci < wanted.size() && layout_ok; ++ci) {
        std::vector<segment> col_segs;
        for (std::size_t g = 0; g < rgs.size() && layout_ok; ++g) {
          auto ps = pieces[ci][g];
          std::sort(ps.begin(), ps.end(),
                    [](auto const& a, auto const& b) { return a.row_offset < b.row_offset; });
          std::int64_t cursor = 0;
          for (auto const& p : ps) {
            auto const off = static_cast<std::int64_t>(p.row_offset);
            if (off < cursor) { layout_ok = false; break; }  // overlapping pages: not one grid
            if (off > cursor) { col_segs.push_back({rg_start[g] + cursor, off - cursor, false, {}}); }
            col_segs.push_back({rg_start[g] + off, static_cast<std::int64_t>(p.num_rows), true,
                                p.slab->view});
            cursor = off + static_cast<std::int64_t>(p.num_rows);
          }
          if (cursor > rg_len[g]) { layout_ok = false; }
          if (layout_ok && cursor < rg_len[g]) {
            col_segs.push_back({rg_start[g] + cursor, rg_len[g] - cursor, false, {}});
          }
        }
        // Neighbouring missing ranges become one read.
        std::vector<segment> merged;
        for (auto const& x : col_segs) {
          if (!x.cached && !merged.empty() && !merged.back().cached &&
              merged.back().start + merged.back().len == x.start) {
            merged.back().len += x.len;
          } else {
            merged.push_back(x);
          }
        }
        for (auto const& x : merged) {
          if (!x.cached) { missing[{x.start, x.len}].push_back(ci); }
        }
        segs[ci] = std::move(merged);
      }
      // One read per distinct missing range, for every column that lacks exactly that range.
      // skip_rows/num_rows address rows across the whole file, which is why row groups are
      // not set here: cuDF does not combine the two.
      std::vector<std::unique_ptr<cudf::table>> reads;
      std::map<std::pair<std::size_t, std::int64_t>, cudf::column_view> read_view;
      std::size_t read_rows = 0;
      ps_peek_us += elapsed_us(t_ps0);
      auto const t_psread0 = std::chrono::steady_clock::now();
      if (layout_ok) {
        for (auto const& [range, cols] : missing) {
          auto o = *split.reader_options;
          std::vector<std::string> names;
          names.reserve(cols.size());
          for (auto const ci : cols) { names.push_back(wanted[ci]); }
          o.set_column_names(names);
          o.set_skip_rows(range.first);
          o.set_num_rows(range.second);
          std::vector<std::unique_ptr<cudf::io::datasource>> src;
          src.push_back(slice.datasource ? cudf::io::datasource::create(slice.datasource.get())
                                         : cudf::io::datasource::create(slice.file_path));
          std::vector<cudf::io::parquet::FileMetaData> md{*slice.file_metadata};
          auto res = cudf::io::read_parquet(std::move(src), std::move(md), o, stream, mr_ref);
          for (std::size_t k = 0; k < res.metadata.schema_info.size(); ++k) {
            for (auto const ci : cols) {
              if (wanted[ci] == res.metadata.schema_info[k].name) {
                read_view[{ci, range.first}] = res.tbl->get_column(static_cast<cudf::size_type>(k)).view();
              }
            }
          }
          read_rows += static_cast<std::size_t>(range.second) * cols.size();
          if (cache_hit_trace_enabled()) { hit_read_bytes += decoded_table_bytes(res.tbl->view(), stream); }
          reads.push_back(std::move(res.tbl));
        }
      }
      if (scan_prep_sync_enabled()) { stream.synchronize(); }
      ps_read_us = elapsed_us(t_psread0);
      auto const t_psasm0 = std::chrono::steady_clock::now();
      std::vector<std::unique_ptr<cudf::column>> assembled;
      bool ok = layout_ok;
      for (std::size_t ci = 0; ci < wanted.size() && ok; ++ci) {
        std::vector<cudf::column_view> parts;
        for (auto const& x : segs[ci]) {
          if (x.cached) {
            parts.push_back(x.view);
            continue;
          }
          auto const it = read_view.find({ci, x.start});
          if (it == read_view.end() || it->second.size() != x.len) {
            ok = false;
            break;
          }
          parts.push_back(it->second);
        }
        if (!ok) { break; }
        assembled.push_back(parts.size() == 1
                              ? std::make_unique<cudf::column>(parts.front(), stream, mr_ref)
                              : cudf::concatenate(parts, stream, mr_ref));
      }
      if (ok) {
        partial_table = std::make_unique<cudf::table>(std::move(assembled));
        partial_hit   = true;
      }
      if (scan_prep_sync_enabled()) { stream.synchronize(); }
      ps_assemble_us = elapsed_us(t_psasm0);
      SIRIUS_LOG_INFO("[fixed-page-cache] partial_serve file='{}' row_groups={} columns={} "
                      "read_ranges={} read_cells={} total_cells={} used={}",
                      slice.file_path,
                      rgs.size(),
                      wanted.size(),
                      missing.size(),
                      read_rows,
                      static_cast<std::size_t>(total_rows) * wanted.size(),
                      ok ? 1 : 0);
    }
  }
  // Cached columns line up with freshly read ones when both went through the SAME
  // predicate. Two cases qualify: neither is filtered (reader_filter_root null), or
  // the cached chunk was produced by exactly this predicate -- the reader is about
  // to apply it again to the columns it reads, so the two halves keep the same rows
  // in the same order. Restricting this to the unfiltered case alone left it unable
  // to help post-filter caching, which is where the misses are.
  if (_scan_manager != nullptr && !all_slices_pruned && split.rg_slices.size() == 1 &&
      !projected_names.empty() && !partial_hit) {
    std::vector<std::string> wanted(projected_names.begin(), projected_names.end());
    auto const sig = reader_filter_root != nullptr ? fixed_page_cache_filter_signature()
                                                   : std::string{};
    auto const conj = reader_filter_root != nullptr ? fixed_page_cache_filter_conjuncts()
                                                    : std::vector<std::string>{};
    // Ask what the cache HAS before asking it for anything. Materializing a column
    // concatenates its pages, which copies every byte, and the decision below throws
    // that away unless the splice is worth doing -- on TPC-H SF50 that was 1,187 of
    // 1,407 scans paying a full copy for nothing.
    auto const t_look0 = std::chrono::steady_clock::now();
    auto const available = _scan_manager->cached_row_group_columns_available(
      split.rg_slices.front().file_path,
      split.rg_slices.front().row_group_indices,
      wanted,
      sig,
      conj);
    std::size_t have = 0;
    for (std::size_t i = 0; i < available.size(); ++i) {
      if (available[i]) { ++have; } else { read_names.push_back(wanted[i]); }
    }
    trace_available = have;
    // Nothing to gain when the cache has none of them, and the all-cached case is
    // left to the existing provider path rather than duplicated here.
    scan_probe_lookup_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t_look0).count();
    // Every column cached is a full hit: build the batch from the cache and skip the read.
    // This used to be excluded here and "left to the existing provider path", but that path
    // only consults the pinned-entry store, so a scan whose every column sat in this store
    // read them all from the file again.
    if (have > 0) {
      auto const t_mat0 = std::chrono::steady_clock::now();
      spliced = _scan_manager->cached_row_group_columns(
        split.rg_slices.front().file_path,
        split.rg_slices.front().row_group_indices,
        wanted,
        stream,
        sig,
        conj);
      if (scan_prep_sync_enabled()) { stream.synchronize(); }
      scan_probe_materialize_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - t_mat0).count();
      // Decide what to read from what the cache actually HANDED OVER, not from what the
      // probe said it had. The two calls take the page store's lock separately, so an
      // eviction in between can drop pages the probe had just counted; trusting the probe
      // leaves those columns out of the read with nothing to supply them. This runs before
      // set_column_names, so a column the cache lost is simply read like any other.
      read_names.clear();
      for (std::size_t i = 0; i < wanted.size(); ++i) {
        if (!spliced[i]) { read_names.push_back(wanted[i]); }
      }
      have = wanted.size() - read_names.size();
      if (read_names.empty() && !_plan->has_partitions()) {
        // Sound without a reader to line up against: a cached page's rows are exactly the
        // rows its filter identity names -- every row of the row group, or the rows that
        // passed this same static predicate (pages from reads that also applied a dynamic
        // filter are never inserted). A dynamic filter on THIS scan only prunes, and
        // post_filter_and_project re-applies the static predicate whenever the state
        // below says it has not run.
        full_hit = true;
      } else if (have == 0 || read_names.empty()) {
        // Nothing left to splice, or nothing left to read beside it. Either way the
        // reader's options are untouched, so it produces every column as it normally
        // would and this scan simply does not use the cache.
        spliced.clear();
      } else {
        splicing = true;
        opts.set_column_names(read_names);
      }
      SIRIUS_LOG_INFO("[fixed-page-cache] column_splice file='{}' row_groups={} cached={} read={}",
                      split.rg_slices.front().file_path,
                      split.rg_slices.front().row_group_indices.size(),
                      have,
                      read_names.size());
    }
  }

  stage_prep_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_stage_begin).count();
  if (scan_cache_trace_enabled() && split.rg_slices.size() == 1) {
    auto const n_cols = projected_names.size();
    auto const served = full_hit ? n_cols : (splicing ? n_cols - read_names.size() : 0);
    SIRIUS_LOG_INFO("[scan-cache-decision] file='{}' row_groups={} columns={} filtered={} "
                    "available={} served={} full_hit={} partial={}",
                    split.rg_slices.front().file_path,
                    split.rg_slices.front().row_group_indices.size(),
                    n_cols,
                    reader_filter_root != nullptr ? 1 : 0,
                    trace_available,
                    partial_hit ? n_cols : served,
                    full_hit ? 1 : 0,
                    partial_hit ? 1 : 0);
  }
  auto const audit_start = std::chrono::steady_clock::now();
  std::unique_ptr<cudf::table> table;
  if (full_hit) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(spliced.size());
    for (auto& cached : spliced) {
      // A column concatenated from several pages belongs to this call alone; move it out.
      // A single-page column is the store's own page and has to be copied.
      if (cached.use_count() == 1) {
        columns.push_back(std::make_unique<cudf::column>(std::move(*cached)));
      } else {
        columns.push_back(std::make_unique<cudf::column>(cached->view(), stream, mr_ref));
      }
    }
    spliced.clear();
    table = std::make_unique<cudf::table>(std::move(columns));
    SIRIUS_LOG_INFO("[fixed-page-cache] full_hit file='{}' row_groups={} columns={} rows={}",
                    split.rg_slices.front().file_path,
                    split.rg_slices.front().row_group_indices.size(),
                    table->num_columns(),
                    table->num_rows());
  } else if (partial_hit) {
    table = std::move(partial_table);
  } else {
    table =
      cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref).tbl;
    if (cache_hit_trace_enabled() && table) {
      hit_read_bytes += decoded_table_bytes(table->view(), stream);
    }
  }
  auto const t_read_end = std::chrono::steady_clock::now();
  scan_probe_read_us =
    std::chrono::duration_cast<std::chrono::microseconds>(t_read_end - audit_start).count();

  if (splicing && table) {
    // Rebuild the projection order: cached column where we have one, otherwise the
    // next column that came back from the reader.
    auto read_cols = table->release();
    if (read_cols.size() != read_names.size()) {
      SIRIUS_LOG_WARN("[fixed-page-cache] column_splice abandoned: reader returned {} of {}",
                      read_cols.size(),
                      read_names.size());
      // Cannot reassemble safely; fall back by re-reading everything.
      return materialize_metadata_to_table(info, mem_space, stream);
    }
    // The availability probe and the materialization take the page store's lock
    // separately, so the store can change in between: an eviction between the two drops
    // pages the probe had just counted, and what comes back is a column built from what
    // survived. `read_names` was already fixed by then, so the missing rows have nobody to
    // supply them, and the table constructor below fails with "Column size mismatch" --
    // seen as 13059667 != 9997497 on ClickBench q22, reachable once the memory-pressure
    // eviction is turned on. Check the splice holds together before trusting it: every
    // cached column must be present and must hold exactly the rows the reader produced.
    // Anything else falls back to re-reading, which costs time and never correctness.
    auto const read_rows = read_cols.empty() ? 0 : read_cols.front()->size();
    std::size_t cached_seen = 0;
    bool splice_intact      = true;
    for (auto const& cached : spliced) {
      if (!cached) { continue; }
      ++cached_seen;
      if (cached->size() != read_rows) {
        SIRIUS_LOG_WARN("[fixed-page-cache] column_splice abandoned: cached column has {} rows, "
                        "reader produced {}",
                        cached->size(),
                        read_rows);
        splice_intact = false;
        break;
      }
    }
    if (splice_intact && cached_seen + read_names.size() != spliced.size()) {
      SIRIUS_LOG_WARN("[fixed-page-cache] column_splice abandoned: {} cached + {} read != {}",
                      cached_seen,
                      read_names.size(),
                      spliced.size());
      splice_intact = false;
    }
    if (!splice_intact) { return materialize_metadata_to_table(info, mem_space, stream); }

    std::vector<std::unique_ptr<cudf::column>> assembled;
    assembled.reserve(spliced.size());
    std::size_t next_read = 0;
    for (auto const& cached : spliced) {
      if (cached && cached.use_count() == 1) {
        // Concatenated for this call alone: take it instead of copying it a second time.
        assembled.push_back(std::make_unique<cudf::column>(std::move(*cached)));
      } else if (cached) {
        assembled.push_back(std::make_unique<cudf::column>(cached->view(), stream, mr_ref));
      } else {
        assembled.push_back(std::move(read_cols[next_read++]));
      }
    }
    table = std::make_unique<cudf::table>(std::move(assembled));
  }
  if (cache_hit_trace_enabled() && table && split.rg_slices.size() >= 1) {
    auto const total = decoded_table_bytes(table->view(), stream);
    SIRIUS_LOG_INFO("[cache-hit] file='{}' total_bytes={} read_bytes={} kind={} "
                    "path=parquet rows={} metric=value_bytes_v2",
                    split.rg_slices.front().file_path, total, std::min(hit_read_bytes, total),
                    full_hit ? "full" : partial_hit ? "partial" : splicing ? "splice" : "miss",
                    table->num_rows());
  }
  auto const audit_end = std::chrono::steady_clock::now();
  scan_probe_assemble_us =
    std::chrono::duration_cast<std::chrono::microseconds>(audit_end - t_read_end).count();
  auto const audit_duration_us =
    std::chrono::duration_cast<std::chrono::microseconds>(audit_end - audit_start).count();
  SIRIUS_LOG_INFO(
    "[scan-audit] parquet_materialize target_gpu={} files={} columns={} row_groups={} "
    "compressed_bytes={} uncompressed_bytes={} output_rows={} output_columns={} split_count={} "
    "duration_us={} lookup_us={} materialize_us={} read_us={} assemble_us={} "
    "lock_wait_us={} stage_prep_us={} full_hit={} filter_us={} ps_peek_us={} ps_read_us={} "
    "ps_assemble_us={}",
    mem_space.get_device_id(),
    join_strings(audit_files, '|'),
    audit_columns,
    audit_row_groups,
    audit_compressed_bytes,
    audit_uncompressed_bytes,
    table ? table->num_rows() : 0,
    table ? table->num_columns() : 0,
    split.rg_slices.size(),
    audit_duration_us,
    scan_probe_lookup_us,
    scan_probe_materialize_us,
    scan_probe_read_us,
    scan_probe_assemble_us,
    _scan_manager ? _scan_manager->page_lock_wait_us() : 0,
    stage_prep_us,
    full_hit ? 1 : 0,
    filter_us,
    ps_peek_us,
    ps_read_us,
    ps_assemble_us);

  // Populating the cache runs after the audit line is emitted, so its cost has never
  // appeared in any scan accounting -- and the gap it leaves is the whole story: with a
  // cache resident GPU_SCAN occupies 2.85s more while its read falls 5.04s.
  // With SIRIUS_SCAN_PREP_SYNC=1, finish the read's GPU work first so populate_us is the copy
  // into pages alone rather than also the wait for the decode to complete.
  if (scan_prep_sync_enabled() && auto_cache && table && !full_hit) {
    auto const t_wait0 = std::chrono::steady_clock::now();
    stream.synchronize();
    SIRIUS_LOG_INFO("[scan-populate] decode_wait_us={}", elapsed_us(t_wait0));
  }
  auto const t_populate0 = std::chrono::steady_clock::now();
  if (auto_cache && table && !full_hit) {
    // reader_filter_root is non-null exactly when opts.set_filter() above ran, i.e.
    // when the rows in `table` are already filtered.
    auto_cache_materialized_table(table->view(),
                                  mem_space,
                                  stream,
                                  /*reader_applied_filter=*/reader_filter_root != nullptr,
                                  reader_applied_dynamic_filter,
                                  split.rg_slices);
  }
  SIRIUS_LOG_INFO("[scan-populate] populate_us={}",
                  std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_populate0)
                    .count());

  // Row-identity virtual columns (LATE_MATERIALIZATION): appended after the data columns, so
  // assemble_scan_output finds them at plan.virtual_batch_position(i). The batch holds every row
  // of every row group read, in order -- the reader filter is off for such scans -- so each
  // row group contributes the run [start, start + rows) of its file's row numbers.
  if (_plan->has_virtual_columns() && table) {
    std::vector<std::unique_ptr<cudf::column>> rn_parts, fi_parts;
    std::int64_t expected = 0;
    for (auto const& slice : split.rg_slices) {
      std::int64_t slice_rows = 0;
      if (slice.file_metadata) {
        auto const& rgs = slice.file_metadata->row_groups;
        std::vector<std::int64_t> start(rgs.size() + 1, 0);
        for (std::size_t i = 0; i < rgs.size(); ++i) {
          start[i + 1] = start[i] + static_cast<std::int64_t>(rgs[i].num_rows);
        }
        for (auto const rg : slice.row_group_indices) {
          auto const r   = static_cast<std::size_t>(rg);
          auto const len = start[r + 1] - start[r];
          cudf::numeric_scalar<std::int64_t> init(start[r], true, stream);
          cudf::numeric_scalar<std::int64_t> step(1, true, stream);
          rn_parts.push_back(
            cudf::sequence(static_cast<cudf::size_type>(len), init, step, stream, mr_ref));
          slice_rows += len;
        }
      }
      auto const it = std::find(_file_paths.begin(), _file_paths.end(), slice.file_path);
      auto const file_index =
        static_cast<std::uint64_t>(it == _file_paths.end() ? 0 : it - _file_paths.begin());
      cudf::numeric_scalar<std::uint64_t> fi(file_index, true, stream);
      fi_parts.push_back(cudf::make_column_from_scalar(
        fi, static_cast<cudf::size_type>(slice_rows), stream, mr_ref));
      expected += slice_rows;
    }
    if (all_slices_pruned) { expected = 0; }
    if (expected != static_cast<std::int64_t>(table->num_rows())) {
      throw sirius::internal_exception(
        "[parquet_gpu_ingestible] file_row_number: batch has " +
        std::to_string(table->num_rows()) + " rows but its row groups hold " +
        std::to_string(expected));
    }
    auto concat_or_empty = [&](std::vector<std::unique_ptr<cudf::column>>& parts,
                               cudf::data_type type) -> std::unique_ptr<cudf::column> {
      if (expected == 0 || parts.empty()) { return cudf::make_empty_column(type); }
      if (parts.size() == 1) { return std::move(parts.front()); }
      std::vector<cudf::column_view> views;
      for (auto const& p : parts) { views.push_back(p->view()); }
      return cudf::concatenate(views, stream, mr_ref);
    };
    auto columns = table->release();
    for (auto const& vc : _plan->virtual_columns) {
      columns.push_back(vc.kind == scan_plan::virtual_column::FILE_ROW_NUMBER
                          ? concat_or_empty(rn_parts, cudf::data_type{cudf::type_id::INT64})
                          : concat_or_empty(fi_parts, cudf::data_type{cudf::type_id::UINT64}));
    }
    table = std::make_unique<cudf::table>(std::move(columns));
  }

  // Hive-partition scans assemble inline here: partition_values are per-split
  // (carried on parquet_split_info) and do not travel to the pipeline-shared
  // post_filter info. Apply the row filter first when pushdown did not, then
  // inject the partition columns and project to the output layout, so the
  // result is fully ROW_FILTERED_AND_PROJECTED and post_filter_and_project is
  // skipped. `sirius_filter_ast` must outlive `exec` — the evaluator borrows it.
  if (_plan->has_partitions()) {
    owning_table_view view{std::move(table)};
    if (!ast_expression.has_value() && _duckdb_filter_expression) {
      auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
      sirius::expression_evaluator exec(sirius_filter_ast.get(), mr_ref, stream);
      auto const data_positions = output_data_positions(*_plan);
      view = data_positions.empty() ? owning_table_view{exec.select(view.view())}
                                    : owning_table_view{exec.select(view.view(), data_positions)};
    }
    auto assembled = assemble_scan_output(*_plan, std::move(view), split.partition_values, stream);
    return op::scan::filtered_table{std::move(assembled),
                                    op::scan::filter_state::ROW_FILTERED_AND_PROJECTED};
  }

  auto const state = ast_expression.has_value() ? op::scan::filter_state::ROW_FILTERED
                                                : op::scan::filter_state::UNFILTERED;
  return op::scan::filtered_table{owning_table_view{std::move(table)}, state};
}

//===----------------------------------------------------------------------===//
// post_filter_and_project — post-decode filter + non-partition projection
//===----------------------------------------------------------------------===//
// Hive-partition scans are fully assembled in materialize_table (it owns the
// per-split partition values) and return ROW_FILTERED_AND_PROJECTED, so they
// never reach here. This path therefore only applies a pending row filter and a
// non-partition projection; partition injection is unreachable.
std::unique_ptr<cudf::table> parquet_gpu_ingestible::post_filter_and_project(
  filtered_table&& input,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());

  // Apply the row filter post-decode when materialization did not — reader-side
  // pushdown was disabled (FLBA-decimal file) or AST translation failed. A
  // ROW_FILTERED / ROW_FILTERED_AND_PROJECTED state means the reader already
  // applied it. `sirius_filter_ast` must outlive `exec` — the evaluator only
  // borrows the AST.
  if (input.state != filter_state::ROW_FILTERED &&
      input.state != filter_state::ROW_FILTERED_AND_PROJECTED && _duckdb_filter_expression) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    sirius::expression_evaluator exec(sirius_filter_ast.get(), mr_ref, stream);
    // With virtual columns, keep every column through the filter: they sit after the data
    // columns and output_data_positions only lists data ones, so a projecting select would cut
    // them off before assemble_scan_output can place them.
    auto const data_positions = output_data_positions(*_plan);
    auto filtered = (data_positions.empty() || _plan->has_virtual_columns())
                      ? exec.select(input.table.view())
                      : exec.select(input.table.view(), data_positions);
    input = filtered_table{owning_table_view{std::move(filtered)}, filter_state::ROW_FILTERED};
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible::post_filter_and_project] Applied duckdb filter expression "
      "post-decode.");
  }

  // Project / reorder the reader's D-order batch to the plan's output layout
  // (non-owning select_columns, no GPU copy). No partitions reach this path, so
  // partition_values is unused. The release below moves the surviving column
  // buffers out.
  auto assembled =
    assemble_scan_output(*_plan, std::move(input.table), /*partition_values=*/{}, stream);
  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::post_filter_and_project] Assembled scan output to plan layout.");
  return assembled.release(stream, mr_ref);
}

//===----------------------------------------------------------------------===//
// materialized_column_order
//===----------------------------------------------------------------------===//
std::vector<std::size_t> parquet_gpu_ingestible::materialized_column_order() const
{
  // The reader materializes columns in _plan->data_columns order (output columns first,
  // pure-filter columns trailing; partition/virtual excluded) — exactly the layout
  // post_filter_and_project's filter refs (batch_position_by_column_id) and output_layout
  // assume. Expose it as primary/storage indices for the pinned-cache path.
  std::vector<std::size_t> order;
  order.reserve(_plan->data_columns.size());
  for (auto const& dc : _plan->data_columns) {
    order.push_back(dc.primary_idx);
  }
  return order;
}

}  // namespace sirius::op::scan
