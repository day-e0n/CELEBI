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

#pragma once

#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "io/datasource_factory.hpp"
#include "io/sirius_datasource.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/split_provider.hpp"
#include "scan_manager/variable_width_page_index.hpp"

// Forward-declare sirius_ioctx via <io/types.hpp> for the gpu_ioctxs map type
// used by prepare_for_query / create_provider_for.
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <io/types.hpp>

#include <atomic>
#include <cstdint>
#include <utility>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}  // namespace cucascade::memory

#include <memory>
#include <list>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::memory {
class topology_index;
}  // namespace sirius::memory

namespace sirius::io {
class sirius_ioctx;
namespace cache {
class buffer_pool;
}  // namespace cache
}  // namespace sirius::io

namespace sirius::op::scan {
class sirius_gpu_scan_operator;
class gpu_ingestible;
}  // namespace sirius::op::scan

namespace sirius::scan_manager {
class load_balancing_scan_batch_coalescer;
}  // namespace sirius::scan_manager

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/// Lightweight descriptor of a pinned table's cache identity + column layout,
/// stored on @ref pinned_entry in place of the read-side ingestible_table_info.
/// Captures only what serving needs — the table's identity (parquet file set OR
/// duckdb catalog/schema/table name), the cached columns (by primary/storage
/// index), and their names (aligned with @c column_ids) for the GPU gather — and
/// owns the match logic that @ref sirius_scan_manager::try_assign_cached_entries consults.
/// One conjunct of a scan's filter, as a value range on a single column.
///
/// The point of holding the predicate this way rather than as its ToString() is
/// reuse across DIFFERENT predicates: cached rows are {r : P(r)}, a query needs
/// {r : Q(r)}, and the cache is safe exactly when Q implies P. For conjunctions of
/// range comparisons that is a containment test per column, which a string compare
/// cannot do -- it only ever matches a byte-identical predicate, so two queries
/// asking for overlapping date windows share nothing.
struct cache_filter_range {
  std::string column_name;  ///< full-schema column name, as the cache keys columns
  bool has_lo{false};
  bool has_hi{false};
  bool lo_inclusive{true};
  bool hi_inclusive{true};
  duckdb::Value lo;
  duckdb::Value hi;
};

/// Whether every row @p consumer selects is one @p producer also selected.
///
/// True when each of the producer's conjuncts is matched by a consumer conjunct on
/// the same column whose range is contained in it. An unanalyzable predicate on
/// either side returns false, which falls back to the byte-identical signature
/// match rather than guessing.
[[nodiscard]] bool filter_ranges_subsume(std::vector<cache_filter_range> const& producer,
                                         bool producer_analyzable,
                                         std::vector<cache_filter_range> const& consumer,
                                         bool consumer_analyzable);

class cache_entry_info {
 public:
  std::vector<std::string> resolved_file_paths;    ///< parquet identity (file set)
  std::string catalog_name;                        ///< duckdb identity: catalog (attach alias)
  std::string schema_name;                         ///< duckdb identity: schema
  std::string table_name;                          ///< duckdb identity: table
  std::string filter_signature;                    ///< non-empty for filter-specific auto caches
  /// The same predicate as @c filter_signature, decomposed into per-column value
  /// ranges. Lets an entry serve a DIFFERENT query whose predicate is strictly
  /// narrower -- see filter_ranges_subsume. Empty with @c filter_analyzable false
  /// when the predicate is not a conjunction of simple range comparisons.
  std::vector<cache_filter_range> filter_ranges;
  bool filter_analyzable{false};
  /// True when the predicate was left out of the cache key and each chunk carries
  /// its own ranges instead (chunk_provenance::filter_ranges), so a consumer clears
  /// chunks one at a time rather than matching one predicate for the whole entry.
  bool chunks_carry_filter_ranges{false};
  duckdb::vector<duckdb::ColumnIndex> column_ids;  ///< cached columns, by primary index
  std::vector<std::string> names;                  ///< aligned with column_ids; gather keys
  /// Footer estimate of each column's whole-table decoded size, aligned with
  /// @c names. Empty when the reader could not supply it. Variable-width paging
  /// uses it to refuse, once and up front, a column that can never be resident
  /// -- see index_variable_width_columns_for_chunk.
  std::vector<std::size_t> projected_column_bytes;
  /// Rows the whole table has, from the parquet footer. Nothing else here can
  /// answer "is this entry complete?" -- pinned_entry::num_rows is a running total
  /// of what happened to be inserted, so a scan that failed halfway leaves a short
  /// entry that reads as valid and silently serves fewer rows. 0 when the reader
  /// could not supply it; completeness is then unknown and the entry is treated as
  /// complete, which is today's behaviour.
  std::size_t table_total_rows{0};
  /// Row groups the scan reads, per file, after stats pruning. An entry is
  /// complete when its chunk provenance covers all of them. Works for filtered
  /// entries too, unlike a row-count comparison. Empty when unknown.
  std::vector<std::pair<std::string, std::unordered_set<int>>> expected_row_groups;

  /// Build the cache descriptor from a read-side ingestible_table_info (parquet
  /// or duckdb-native): captures the format's identity, the kept @c column_ids,
  /// and the @c column_ids-aligned column names.
  [[nodiscard]] static cache_entry_info from(const op::scan::ingestible_table_info& info);

  /// Gather projection (positions into @c column_ids) that lets this cached entry
  /// serve @p other — matching identity (same parquet file set / same duckdb
  /// catalog.schema.table) AND a superset of @p other's requested columns. The projection
  /// reproduces @p other's requested column order. Empty when this entry cannot
  /// serve @p other (different format, identity, or a missing column).
  [[nodiscard]] std::vector<std::size_t> can_serve_with_columns(
    const op::scan::ingestible_table_info& other) const;

  /// Column names in @c column_ids order — the keys @c data_batches_by_column uses.
  [[nodiscard]] const std::vector<std::string>& column_names() const { return names; }
};

// wdy start
enum class fixed_width_page_stat_kind : uint8_t {
  none,
  signed_int,
  unsigned_int,
  floating,
};

enum class fixed_width_page_state : uint8_t {
  resident,
  evicted,
};

struct fixed_width_page_directory_key {
  std::string table_name;
  std::string file_path;
  std::string column_name;
  std::size_t chunk_index{0};
  std::size_t page_index{0};
  int device_id{-1};
};

struct fixed_width_page_stats {
  bool valid{false};
  bool has_null{false};
  fixed_width_page_stat_kind kind{fixed_width_page_stat_kind::none};
  int64_t min_signed{0};
  int64_t max_signed{0};
  uint64_t min_unsigned{0};
  uint64_t max_unsigned{0};
  double min_floating{0.0};
  double max_floating{0.0};
};

/**
 * @brief Logical page metadata for a fixed-width GPU column chunk.
 *
 * The first prototype keeps physical ownership in the existing cudf::column
 * chunks and records page boundaries over those buffers. A later scan path can
 * use this as the lookup unit for table/column/chunk/page reuse without first
 * changing cudf buffer ownership.
 */
struct fixed_width_column_page {
  fixed_width_column_page() = default;

  fixed_width_column_page(fixed_width_column_page const& other)
    : key(other.key),
      state(other.state),
      chunk_index(other.chunk_index),
      page_index(other.page_index),
      global_row_offset(other.global_row_offset),
      row_offset(other.row_offset),
      num_rows(other.num_rows),
      byte_offset(other.byte_offset),
      num_bytes(other.num_bytes),
      element_size_bytes(other.element_size_bytes),
      type_id(other.type_id),
      memory_space(other.memory_space),
      admission_score(other.admission_score),
      access_count(other.access_count),
      last_access_tick(other.last_access_tick.load(std::memory_order_relaxed)),
      active_reader_count(other.active_reader_count.load(std::memory_order_relaxed)),
      owned_column(other.owned_column),
      stats(other.stats)
  {}

  fixed_width_column_page& operator=(fixed_width_column_page const& other)
  {
    if (this == &other) { return *this; }
    key                = other.key;
    state              = other.state;
    chunk_index        = other.chunk_index;
    page_index         = other.page_index;
    global_row_offset  = other.global_row_offset;
    row_offset         = other.row_offset;
    num_rows           = other.num_rows;
    byte_offset        = other.byte_offset;
    num_bytes          = other.num_bytes;
    element_size_bytes = other.element_size_bytes;
    type_id            = other.type_id;
    memory_space       = other.memory_space;
    admission_score    = other.admission_score;
    access_count       = other.access_count;
    last_access_tick.store(other.last_access_tick.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    active_reader_count.store(other.active_reader_count.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    owned_column = other.owned_column;
    stats        = other.stats;
    return *this;
  }

  fixed_width_column_page(fixed_width_column_page&& other) noexcept
    : key(std::move(other.key)),
      state(other.state),
      chunk_index(other.chunk_index),
      page_index(other.page_index),
      global_row_offset(other.global_row_offset),
      row_offset(other.row_offset),
      num_rows(other.num_rows),
      byte_offset(other.byte_offset),
      num_bytes(other.num_bytes),
      element_size_bytes(other.element_size_bytes),
      type_id(other.type_id),
      memory_space(other.memory_space),
      admission_score(other.admission_score),
      access_count(other.access_count),
      last_access_tick(other.last_access_tick.load(std::memory_order_relaxed)),
      active_reader_count(other.active_reader_count.load(std::memory_order_relaxed)),
      owned_column(std::move(other.owned_column)),
      stats(other.stats)
  {}

  fixed_width_column_page& operator=(fixed_width_column_page&& other) noexcept
  {
    if (this == &other) { return *this; }
    key                = std::move(other.key);
    state              = other.state;
    chunk_index        = other.chunk_index;
    page_index         = other.page_index;
    global_row_offset  = other.global_row_offset;
    row_offset         = other.row_offset;
    num_rows           = other.num_rows;
    byte_offset        = other.byte_offset;
    num_bytes          = other.num_bytes;
    element_size_bytes = other.element_size_bytes;
    type_id            = other.type_id;
    memory_space       = other.memory_space;
    admission_score    = other.admission_score;
    access_count       = other.access_count;
    last_access_tick.store(other.last_access_tick.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    active_reader_count.store(other.active_reader_count.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    owned_column = std::move(other.owned_column);
    stats        = other.stats;
    return *this;
  }

  fixed_width_page_directory_key key;
  fixed_width_page_state state{fixed_width_page_state::resident};
  std::size_t chunk_index{0};
  std::size_t page_index{0};
  std::size_t global_row_offset{0};
  std::size_t row_offset{0};
  std::size_t num_rows{0};
  std::size_t byte_offset{0};
  std::size_t num_bytes{0};
  std::size_t element_size_bytes{0};
  cudf::type_id type_id{cudf::type_id::EMPTY};
  cucascade::memory::memory_space* memory_space{nullptr};
  /// Retained for admission diagnostics; eviction itself is plain LRU.
  double admission_score{0.0};
  /// Lightweight reuse counter retained for diagnostics.
  std::size_t access_count{0};
  /// Monotonic logical timestamp for LRU eviction. Larger means more recently used.
  mutable std::atomic<std::uint64_t> last_access_tick{0};
  /// Number of page-backed scan providers currently allowed to read this page.
  /// Eviction skips active pages so replacement never invalidates an in-flight scan input.
  mutable std::atomic<std::uint32_t> active_reader_count{0};
  /// Optional page-owned storage. When set, this page can be materialized
  /// without retaining the original full cuDF column chunk.
  std::shared_ptr<cudf::column> owned_column;
  fixed_width_page_stats stats;
};

struct fixed_width_page_directory_metrics {
  std::size_t resident_pages{0};
  std::size_t resident_bytes{0};
  std::size_t stats_pages{0};
  std::size_t evicted_pages{0};
  std::size_t eviction_count{0};
};

struct fixed_width_page_directory_entry {
  std::string column_name;
  std::size_t page_ordinal{0};
};

/// O(1) lookup aid: for a given (column, chunk_index), the contiguous run of
/// pages that chunk contributed to fixed_width_pages_by_column[column] --
/// [page_start_index, page_start_index + page_count). Combined with
/// rows_per_page, a page covering a given in-chunk row_offset is found by
/// direct arithmetic instead of scanning the page list. page_count == 0 means
/// no pages were recorded for this chunk (e.g. a column merged in after this
/// chunk_index already existed for other columns), matching a lookup miss.
struct fixed_width_chunk_page_span {
  std::size_t page_start_index{0};
  std::size_t page_count{0};
  std::size_t rows_per_page{0};
};

/**
 * @brief A single pinned-table entry, keyed by table name in the scan_manager.
 *
 * Stores the column projection captured at pin time (so the scan side knows
 * which columns the user pinned) along with the data batches making up the
 * pinned table. The vector may be empty until splits are populated.
 */
/// Which parquet row groups a cached chunk actually holds.
///
/// chunk_index is an arrival counter (`entry.chunk_memory_spaces.size()` at insert
/// time) and a page's global_row_offset is a running total in that same arrival
/// order -- neither names a position in the table. A COMPLETE entry does not care:
/// holding every chunk makes the concatenation a permutation of the table, and
/// every operator above the scan is order-insensitive. A PARTIAL entry does care,
/// because serving part of a scan from cache means asking parquet for the exact
/// complement, and without this there is no way to name it.
///
/// Keyed by file path, never by a flat index: chunk_index == row_group_index holds
/// only while a single row group exceeds the coalescer's byte cap, which is a
/// tuning parameter rather than an invariant.
struct chunk_provenance {
  std::vector<std::pair<std::string, std::vector<cudf::size_type>>> slices;
  std::size_t num_rows{0};
  /// Row count of each row group in @c slices, flattened in the same order.
  /// Empty when the producer could not supply it (non-parquet readers), in which
  /// case page cutting falls back to a plain byte grid over the whole chunk.
  ///
  /// Needed because a page must not straddle a row-group boundary: the residual
  /// path reads whole row groups, so a page spanning two of them cannot be
  /// dropped or kept as a unit. Cutting on the boundary also removes the
  /// remainder waste that a chunk-wide grid leaves -- measured 12.9% on TPC-H's
  /// 9,962,958-row groups and 31.1% on ClickBench's 10,000,000-row ones.
  std::vector<std::size_t> row_group_rows;
  /// The predicate that produced THIS chunk's rows, as per-column value ranges.
  /// Carried per chunk rather than per entry so one entry can hold chunks filtered
  /// differently: the cache identity then needs no filter in it, and a consumer
  /// picks the chunks whose predicate its own is narrower than.
  std::vector<cache_filter_range> filter_ranges;
  bool filter_analyzable{false};
  /// The predicate's text, kept alongside the ranges. A predicate that is not a
  /// conjunction of range comparisons -- `SearchPhrase <> ''`, `contains(URL,...)`,
  /// which is most of ClickBench -- has no ranges to compare, and without this an
  /// identical query could no longer reuse its own chunk: reuse fell 45 -> 3.
  std::string filter_signature;
};

/// One cached column of one row group -- the unit the cache is addressed by.
///
/// Replaces the entry as the thing a scan looks up. An entry bundled pages under a
/// name and made reuse all-or-nothing across everything in the bundle: one absent
/// column, or one predicate that did not match byte for byte, and the whole bundle
/// went unused. Measured on ClickBench: entries held a median of 6 MB -- less than
/// one 16 MB page -- so the bundling bought nothing and cost every reuse decision.
///
/// Addressed by (file, column, row group). Nothing above that: no projection, no
/// predicate. The predicate travels WITH the page instead, as the signature that
/// produced it plus its value ranges, so a scan can ask "are these rows a superset
/// of what I need?" per page rather than per bundle.
/// Key for @ref cached_page. Kept as a struct rather than a packed string so the
/// parts stay inspectable in logs and in eviction, which works per file.
struct cached_page_key {
  /// File path and column name as interned ids, not strings. The page loop below
  /// runs once per page of every column of every row group a scan touches, and
  /// hashing a 60-character path there cost 4.10s of TPC-H SF50's scan time once
  /// row groups were cut into 16MiB pages. Interning moves that to one hash per
  /// column per scan and leaves four ints in the inner loop.
  int file_id{-1};
  int column_id{-1};
  int row_group{0};
  /// Which page within that row group. A row group is cut into pages of about
  /// fixed_width_page_size_bytes() so the cache's unit of residency and eviction
  /// is a page, not a whole row group -- a ClickBench row group of URL is 0.9GiB,
  /// 15% of a 6GB budget, and evicting one throws all of it away.
  int page_index{0};

  [[nodiscard]] bool operator==(cached_page_key const& other) const = default;
};

struct cached_page {
  std::shared_ptr<cudf::column> data;  ///< one page of this column within one row group
  std::size_t num_rows{0};
  /// How many pages the row group was cut into. The serve path needs it to tell a
  /// complete run from a truncated one: pages 0..2 present with page 3 evicted
  /// would otherwise look whole and silently hand back a column missing its tail.
  std::size_t pages_in_row_group{1};
  /// Position in @ref _lru. Touching a page splices it to the back in O(1), and
  /// eviction pops the front -- so neither path has to scan or sort the store.
  std::list<cached_page_key>::iterator lru_it{};
  std::size_t num_bytes{0};
  fixed_width_page_stats stats;  ///< min/max, for skipping a page a predicate cannot match
  /// Predicate the rows came through. Empty means unfiltered, which satisfies any
  /// request; otherwise a request is served only when it is narrower -- by exact
  /// signature, or by range containment.
  std::string filter_signature;
  /// The same predicate as its AND-ed parts. A request whose parts are a superset
  /// of these selects a subset of these rows, so the page serves it -- the test
  /// the range comparison cannot make for `<>` or LIKE.
  std::vector<std::string> filter_conjuncts;
  std::vector<cache_filter_range> filter_ranges;
  bool filter_analyzable{false};
  std::uint64_t last_access_tick{0};
};


struct cached_page_key_hash {
  [[nodiscard]] std::size_t operator()(cached_page_key const& key) const noexcept
  {
    // Four ints packed into one 64-bit word, then mixed. No string hashing on a
    // path this hot.
    auto const packed = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.file_id)) << 48) ^
                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.column_id)) << 32) ^
                        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.row_group)) << 16) ^
                        static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.page_index));
    return std::hash<std::uint64_t>{}(packed * 0x9e3779b97f4a7c15ULL);
  }
};

struct pinned_entry {
  /// Cache identity + column layout for this pinned table. Drives the cache-hit
  /// match (@ref cache_entry_info::can_serve_with_columns) and the per-column
  /// gather; replaces the heavyweight read-side ingestible_table_info.
  cache_entry_info cache_info;

  /// GPU-tier storage: one chunk vector per pinned column name. Populated by
  /// @ref sirius_scan_manager::insert_pinned_entry. Empty when @ref tier is HOST.
  std::unordered_map<std::string, std::vector<std::shared_ptr<cudf::column>>>
    data_batches_by_column;
  /// Prototype fixed-width page index. Keyed by column name; each entry is a
  /// logical page over the corresponding cudf column chunk in
  /// data_batches_by_column. Variable-width and nested columns are omitted.
  std::unordered_map<std::string, std::vector<fixed_width_column_page>> fixed_width_pages_by_column;
  /// Directory lookup keyed by table/file/column/chunk/page/device. Values point
  /// back into fixed_width_pages_by_column without owning page storage.
  std::unordered_map<std::string, fixed_width_page_directory_entry> fixed_width_page_directory;
  /// Per-column, chunk_index-indexed page spans (see fixed_width_chunk_page_span) --
  /// lets find_covering_fixed_page locate the page covering a given row by direct
  /// arithmetic instead of scanning fixed_width_pages_by_column. Populated by
  /// index_fixed_width_column_pages alongside the page list itself.
  std::unordered_map<std::string, std::vector<fixed_width_chunk_page_span>>
    fixed_width_chunk_page_spans;
  /// Target page size used when fixed_width_pages_by_column was built.
  std::size_t fixed_width_page_size_bytes{0};
  /// Prototype variable-width (STRING) page index, entirely separate from the
  /// fixed-width members above -- see variable_width_page_index.hpp. Keyed by
  /// column name; each entry is this column's pages for one chunk, indexed by
  /// chunk_index (parallel in spirit to fixed_width_chunk_page_spans, but one
  /// full variable_width_chunk_page_index per chunk since page row-counts
  /// aren't a single constant to summarize).
  std::unordered_map<std::string, std::vector<variable_width_chunk_page_index>>
    variable_width_pages_by_column;
  /// Target page size used when variable_width_pages_by_column was built.
  std::size_t variable_width_page_size_bytes{0};
  /// Lightweight page-directory accounting for the current prototype. Pages are
  /// still physically owned by cudf column chunks, but this is the metadata
  /// surface that scan reuse and future page admission/eviction build on.
  fixed_width_page_directory_metrics fixed_width_page_metrics;
  /// Per-chunk memory space placement. Parallel to the inner vectors of
  /// data_batches_by_column: chunk_memory_spaces[i] is the memory_space*
  /// for every column's chunk at index i. All columns at chunk index i
  /// share the same memory_space because they came from the same
  /// chunked_parquet_reader::read_chunk() call.
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
  /// Parallel to chunk_memory_spaces: the row groups each chunk came from.
  std::vector<chunk_provenance> chunk_provenance_by_index;
  /// Columns stored as dictionary CODES rather than as strings. Their pages live in
  /// fixed_width_pages_by_column like any int32 column; the keys needed to turn a
  /// code back into a string live in the scan manager's shared dictionary store,
  /// once per (identity, column) rather than once per chunk.
  std::unordered_set<std::string> dictionary_encoded_columns;
  /// HOST-tier storage: one host_data_representation per chunk, each holding all
  /// pinned columns. The cached_split_provider slices these by column index when
  /// serving a particular scan. Populated by @ref insert_pinned_entry_host.
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;
  /// Tier the pinned data resides in. Drives which storage member above is used
  /// and which cached_split_provider variant @ref create_provider_for builds.
  cucascade::memory::Tier tier{cucascade::memory::Tier::GPU};
  /// Memory space the pinned data resides in. Captured at pin time so the
  /// cached_split_provider can wrap copied tables as data_batch instances.
  cucascade::memory::memory_space* memory_space{nullptr};
  /// Total number of rows across all pinned chunks. Used by insert_pinned_entry
  /// to decide whether a re-insert merges into the existing entry (same row
  /// count → add unique columns) or replaces it (different row count).
  std::size_t num_rows{0};
};

/**
 * @brief Bind-time result of @ref sirius_scan_manager::describe_parquet.
 *
 * Carries the column types and names a parquet file's footer yields, ready to
 * be copied into a DuckDB table function's bind out-parameters, plus the total
 * object size in bytes.
 */
struct parquet_bind_result {
  duckdb::vector<duckdb::LogicalType> return_types;
  duckdb::vector<std::string> names;
  std::size_t object_size{0};
  std::size_t total_num_rows{0};
};

/**
 * @brief Manages scan-side preparation for a query.
 *
 * The scan manager owns a configurable-size thread pool and is given a chance
 * to set up per-scan state before a query runs (via prepare_for_query).
 */
/// Page metadata plus NON-OWNING handles to the buffers, for sharing one column
/// chunk's pages across the several cache entries whose projections all contain
/// that column. Owning the buffers here would make eviction a no-op:
/// apply_global_variable_width_page_budget resets the entries' owned_column, and a
/// strong reference in the store would keep the allocation alive while the sweep
/// reports the bytes as freed -- the same "eviction that cannot move its own
/// metric" failure as the memory-pressure loop. Weak handles mean the last real
/// holder still frees, and a stale row simply fails to lock and is dropped.
struct shared_variable_pages {
  variable_width_chunk_page_index index;             ///< pages with owned_column left null
  std::vector<std::weak_ptr<cudf::column>> buffers;  ///< parallel to index.pages
};

class sirius_scan_manager {
 public:
  /**
   * @brief Construct a new scan manager.
   *
   * The scan_manager owns a single io_context (uring_ioctx) and optionally
   * an S3 backend and a prefetch buffer pool, all created from @p config.
   *
   * @param config Scan-manager configuration (thread pool + sirius_datasource toggle).
   * @param reservation_manager Memory reservation manager for GPU memory.
   * @param topology_index Hardware GPU/NUMA topology index.  Drives round-robin
   *        GPU assignment for scans and is forwarded to the prefetching cache.
   */
  sirius_scan_manager(const scan_manager_config& config,
                      cucascade::memory::memory_reservation_manager& reservation_manager,
                      std::shared_ptr<const sirius::memory::topology_index> topology_index);

  ~sirius_scan_manager();

  // Non-copyable and non-movable
  sirius_scan_manager(const sirius_scan_manager&)            = delete;
  sirius_scan_manager& operator=(const sirius_scan_manager&) = delete;
  sirius_scan_manager(sirius_scan_manager&&)                 = delete;
  sirius_scan_manager& operator=(sirius_scan_manager&&)      = delete;

  using ingestible_table_info = op::scan::ingestible_table_info;

  /// \brief Per-entry admission ceiling for the fixed-width page cache, in bytes
  /// (0 = no limit; defaults to half the per-GPU budget).
  ///
  /// Exposed so scan-side code can size a prospective entry from its parquet
  /// footer and skip it before decoding, instead of building the copy and having
  /// it erased by the same limit after the fact.
  [[nodiscard]] static std::size_t fixed_page_admission_limit_bytes();

  /// The fixed-width page cache budget for one GPU, in bytes.
  ///
  /// Exposed so the scan side can weigh a table against the cache before deciding
  /// to populate it -- see parquet_gpu_ingestible's dynamic-filter size gate.
  [[nodiscard]] static std::size_t fixed_page_cache_budget_bytes();

  /// \brief Whether the variable-width (STRING) page index is enabled.
  ///
  /// Exposed so scan-side admission can tell that a batch of nothing but STRING
  /// columns is still cacheable, rather than refusing it for having no
  /// fixed-width column.
  [[nodiscard]] static bool variable_width_page_cache_is_enabled();

  /// \brief Whether one cache entry per (file, filter) holds the union of every
  /// projection's columns, instead of one entry per projection.
  ///
  /// Exposed so the scan-side name builder can drop the column set from the cache
  /// identity, which is what makes the merge possible.
  [[nodiscard]] static bool column_keyed_cache_is_enabled();

  /// \brief Prepare per-scan state for the given query.
  ///
  /// Walks @p query 's pipelines in scan-operator order. For each GPU parquet
  /// scan source, the factory builds a split_provider from the operator's
  /// scan_info, installs a fresh split_connector on the operator, and stores
  /// the provider in a map keyed by the operator. A driver thread then runs
  /// the providers SEQUENTIALLY in registration order: provider[0] starts,
  /// when its future completes provider[1] starts, and so on. Consumers (the
  /// gpu scan operators) block in split_connector::get_next_split until splits
  /// arrive or the connector is closed, so no separate wake-up channel is
  /// needed.
  ///
  /// @param query        The query whose scan operators must be prepared.
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief Clear the providers map and join the driver thread if it is
  ///        still running.
  void reset();

  /// \brief Start the worker thread pool. Idempotent.
  void start();

  /// \brief Stop the worker thread pool and the driver. Idempotent.
  void stop();

  /// \brief Pin (or extend) the GPU-tier entry for a table.
  ///
  /// Releases the columns of each input @p data_tables into the entry's per-column
  /// map, keyed by the column names carried in @p cache_info (the i-th column of
  /// every table is appended to @c data_batches_by_column[cache_info.column_names()[i]]).
  /// Tables become empty after this call.
  ///
  /// Re-insert semantics (keyed by @p name):
  ///   - If no entry exists for @p name, a fresh one is created.
  ///   - If an entry exists and its @c num_rows equals the new total row count, the
  ///     incoming columns whose names are not already present are merged in
  ///     (duplicate columns are dropped), and the entry's @c cache_info is extended
  ///     to the union of pinned columns so later cache-hit matching can serve them.
  ///     The existing cache identity is preserved; the merge requires the incoming
  ///     @p chunk_memory_spaces to be identical to the existing entry's and rejects
  ///     any mismatch.
  ///   - If row counts differ, the existing entry is dropped and replaced. (An
  ///     n_rows-capped "partial" pin therefore never merges with a full pin of the
  ///     same table, since their row counts differ.)
  ///
  /// \param name                  Table name key.
  /// \param cache_info            Cache identity (parquet file set or duckdb
  ///                              catalog.schema.table) plus the cached columns by
  ///                              primary index and their @c column_ids-aligned names;
  ///                              drives later cache-hit matching and the per-column gather.
  /// \param data_tables           Cudf tables produced by chunked reads, one per chunk
  ///                              (may be empty). Each table's column count MUST equal
  ///                              the number of columns described by @p cache_info.
  /// \param chunk_memory_spaces   Per-chunk memory space placement; size MUST equal
  ///                              data_tables.size() (the value at index i is shared by
  ///                              all columns at chunk i).
  void insert_pinned_entry(const std::string& name,
                           cache_entry_info cache_info,
                           std::vector<std::unique_ptr<cudf::table>> data_tables,
                           std::vector<cucascade::memory::memory_space*> chunk_memory_spaces);

  /// Populate an automatic fixed-page entry directly from a materialized GPU
  /// table view. This skips the transient full-table cache copy used by
  /// insert_pinned_entry and stores only owned fixed-width pages.
  /// \return Number of new fixed-width pages actually added (0 means not
  ///         populated -- rejected, skipped, or a detected duplicate).
  ///
  /// @param fixed_width_pre_rejected  The caller has already determined (from the
  ///        parquet footer) that this entry's fixed-width columns cannot fit under
  ///        fixed_page_admission_limit_bytes(). Their pages are then skipped
  ///        outright -- no copy is made and no admission check runs -- while STRING
  ///        columns still populate the variable-width page index, which has its own
  ///        independent budget. Without this split, a fixed-width size verdict also
  ///        silently disables variable-width caching for the table (measured on SF50:
  ///        lineitem's variable-width indexing went 61 -> 0). The resulting entry
  ///        holds nullptr chunks for its fixed-width columns, which is the same shape
  ///        SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0 already produces.
  [[nodiscard]] std::size_t insert_fixed_page_entry_from_view(
    const std::string& name,
    cache_entry_info cache_info,
    cudf::table_view view,
    cucascade::memory::memory_space& memory_space,
    rmm::cuda_stream_view stream,
    bool fixed_width_pre_rejected = false,
    chunk_provenance provenance   = {});

  /// \brief Pin the host-tier entry for a table.
  ///
  /// Each entry in @p host_chunks describes one batch's worth of pinned data
  /// (covering all pinned columns) as a host_data_representation. The
  /// cached_split_provider built from this entry slices each chunk by column
  /// index at scan time. This path always REPLACES any existing entry for @p name
  /// — there is no per-column merge analog to the GPU path because the
  /// chunk-vs-column dimensions are flipped (each chunk already holds every column).
  ///
  /// \param name          Table name key.
  /// \param cache_info    Cache identity plus the cached columns and their
  ///                      @c column_ids-aligned names (the i-th column in each
  ///                      host_data_representation corresponds to
  ///                      @c cache_info.column_names()[i]); drives cache-hit matching.
  /// \param host_chunks   One host_data_representation per emitted batch.
  /// \param memory_space  Representative host memory space the chunks reside in
  ///                      (metadata only; each chunk carries its own per-GPU
  ///                      NUMA-local memory_space).
  void insert_pinned_entry_host(
    const std::string& name,
    cache_entry_info cache_info,
    std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
    cucascade::memory::memory_space& memory_space);

  /// \brief Remove the pinned entry for @p name. No-op if absent.
  void remove_pinned_entry(const std::string& name);

  void visit_pinned_entries(
    const std::function<bool(std::string_view, const pinned_entry&)>& visitor) const;

  parquet_bind_result describe_parquet(std::string const& uri);

  /// \brief Process-wide ioctx used to mint @c sirius_datasource instances.
  ///        Returns nullptr when the manager was configured with
  ///        @c use_sirius_datasource=false.
  [[nodiscard]] sirius::io::sirius_ioctx* io_ctx() const noexcept { return _io_ctx.get(); }

  [[nodiscard]] std::shared_ptr<sirius::io::sirius_datasource> create_datasource(
    std::string_view path);

  /// Cached data for @p column_names covering exactly @p row_groups of @p file_path,
  /// concatenated in that order; an entry is nullptr when the column is not cached
  /// for every one of them. Lets a scan read only the columns it is MISSING and
  /// splice the cached ones in, instead of the all-or-nothing rule where one absent
  /// column sends every column back to parquet -- 539 of 602 cache misses on
  /// ClickBench were exactly that.
  ///
  /// Only UNFILTERED cached data qualifies: a filtered entry holds a subset of the
  /// row group's rows and cannot be pasted beside freshly read columns that hold
  /// all of them.
  /// Store one row group's worth of @p column_names from @p view into the flat page
  /// store. @p row_group_rows says how the view's rows divide among @p row_groups,
  /// so each page holds exactly one row group.
  void insert_pages_from_view(std::string const& file_path,
                              std::vector<cudf::size_type> const& row_groups,
                              std::vector<std::size_t> const& row_group_rows,
                              std::vector<std::string> const& column_names,
                              cudf::table_view const& view,
                              std::string const& filter_signature,
                              std::vector<std::string> filter_conjuncts,
                              std::vector<cache_filter_range> filter_ranges,
                              bool filter_analyzable,
                              cucascade::memory::memory_space& space,
                              rmm::cuda_stream_view stream);

  /// Bytes the flat page store holds, and how many pages.
  [[nodiscard]] std::pair<std::size_t, std::size_t> page_store_size() const;

  /// Throw the whole page store away and report how many bytes that returned. Called when a
  /// pipeline task has actually run out of device memory: at that point the cache is holding
  /// memory a running query needs, and every page in it is by definition reconstructible from
  /// the file. Handing back only the configured floor's shortfall is not enough there -- it
  /// was measured returning 0.22 GB against a query that needed gigabytes, after which the
  /// pressure sweep's own circuit breaker concluded eviction was useless and backed off.
  std::size_t drop_all_pages();

  /// Microseconds probes have spent waiting for the page store's mutex, since process start.
  [[nodiscard]] std::int64_t page_lock_wait_us() const
  {
    return _page_lock_wait_us.load(std::memory_order_relaxed);
  }

  /// Which of @p column_names the cache can serve for these row groups, without
  /// materializing any of them.
  ///
  /// cached_row_group_columns concatenates a column's pages into one column, which
  /// allocates and copies every byte -- and its caller throws the result away
  /// unless the splice is worth doing (some columns cached AND some still to
  /// read). Cutting row groups into 16MiB pages made that concatenate real work
  /// where a whole-row-group page had been a single move, and TPC-H SF50 pays it
  /// on 1,187 of 1,407 scans that end up not splicing at all: +1.76s of scan time.
  /// Ask this first, decide, then materialize only if the answer is yes.
  [[nodiscard]] std::vector<bool> cached_row_group_columns_available(
    std::string const& file_path,
    std::vector<cudf::size_type> const& row_groups,
    std::vector<std::string> const& column_names,
    std::string const& required_filter_signature = {},
    std::vector<std::string> const& required_filter_conjuncts = {});

  [[nodiscard]] std::vector<std::shared_ptr<cudf::column>> cached_row_group_columns(
    std::string const& file_path,
    std::vector<cudf::size_type> const& row_groups,
    std::vector<std::string> const& column_names,
    rmm::cuda_stream_view stream,
    std::string const& required_filter_signature = {},
    std::vector<std::string> const& required_filter_conjuncts = {});

 private:
  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  /// \brief Attach a cached batch_provider to @p op if a pinned entry can serve
  ///        it. Returns true when a cache hit was assigned (the caller then skips
  ///        the disk-reading split_provider for this operator).
  /// @param residual_out  Set to true when the cache covers the scan only partly,
  ///        meaning the caller must still build a split_provider for the row groups
  ///        the cache does not hold.
  bool try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op,
                                 bool* residual_out = nullptr);

  /// Drop LRU fixed pages when device free memory is below the configured floor.
  void evict_fixed_pages_for_memory_pressure(std::string_view reason);

  /// Resolve the ioctx that should serve @p path (normalized internally, so callers
  /// — including the scan resolver — may pass a raw `file://` / `s3://` URI),
  /// building it once per backend on first use.  Routes by path through the registry
  /// so an `s3://` URI reaches the rest_ioctx even when the local default `_io_ctx`
  /// is uring/kvikio.  Returns nullptr when no backend supports the path.
  std::shared_ptr<sirius::io::sirius_ioctx> ioctx_for_path(std::string_view path);

  scan_manager_config _config;
  cucascade::memory::memory_reservation_manager& _reservation_manager;
  /// Hardware GPU/NUMA topology, shared with the prefetching cache.  Source of
  /// the GPU id set fed to the round-robin scan-balancing strategy.
  std::shared_ptr<const sirius::memory::topology_index> _topology_index;
  exec::static_thread_pool _thread_pool;
  std::unique_ptr<exec::scoped_dispatcher> _dispatcher;
  std::shared_ptr<sirius::io::sirius_ioctx> _io_ctx;
  /// Lazily-built per-backend ioctxs for path-routed datasources (e.g. an s3://
  /// rest_ioctx alongside the local uring/kvikio `_io_ctx`).  Built exactly once
  /// per type: `_routed_io_ctxs_build_mtx` serializes construction (reactor
  /// threads + cache allocation happen outside the map mutex), while
  /// `_routed_io_ctxs_mtx` guards only map lookup/insert; drained + torn down
  /// in the dtor.
  std::mutex _routed_io_ctxs_build_mtx;
  std::mutex _routed_io_ctxs_mtx;
  std::unordered_map<sirius::io::io_context_type, std::shared_ptr<sirius::io::sirius_ioctx>>
    _routed_io_ctxs;
  std::unordered_map<op::scan::sirius_gpu_scan_operator*, std::unique_ptr<split_provider>>
    _providers_by_op;
  std::vector<op::scan::sirius_gpu_scan_operator*> _scan_op_order;
  mutable std::mutex _pinned_entries_mutex;
  std::unordered_map<std::string, pinned_entry> _pinned_entries;
  /// The flat page store: (file, column, row group) -> one cached column. This is
  /// what the auto page cache is, now that the entry is gone from the lookup path.
  std::unordered_map<cached_page_key, cached_page, cached_page_key_hash> _pages;
  /// Least-recently-used order, oldest at the front. Kept as a list so a touch is
  /// a splice and an eviction is a pop, both O(1). The previous pass rebuilt a
  /// vector of every page and sorted it on EVERY insert -- O(n log n) each time,
  /// and n grew tenfold when row groups were cut into 16MiB pages, which turned
  /// TPC-H SF50 from -3.1% into +2.3% against its own baseline.
  std::list<cached_page_key> _lru;
  /// Microseconds probes have spent waiting for `_pages_mutex`, summed across threads.
  std::atomic<std::int64_t> _page_lock_wait_us{0};
  /// String -> small int for cached_page_key. Grows only; ids stay valid.
  std::unordered_map<std::string, int> _intern;
  [[nodiscard]] int intern_id(std::string const& s);        ///< inserts if absent
  [[nodiscard]] int intern_lookup(std::string const& s) const;  ///< -1 when absent
  /// Running total of resident page bytes, maintained on insert and eviction so
  /// the budget check does not have to sum the whole store.
  std::size_t _pages_bytes{0};
  mutable std::mutex _pages_mutex;
  std::uint64_t _page_tick{0};
  /// Backoff for the device-pressure check -- see where it is applied.
  std::size_t _page_pressure_skip{0};
  /// Set when a pipeline task ran out of device memory and the store was dropped; cleared by
  /// reset(), which runs per query. Dropping alone does not help -- the next batch of the same
  /// query simply refills the store and it runs out again, measured as 305 drop-and-refill
  /// cycles on ClickBench q24 before the retry budget ran out. A query that has already hit
  /// the wall gets no more caching; the next one starts clean.
  std::atomic<bool> _admission_suspended{false};
  std::size_t _page_pressure_free_before{0};

  /// Canonical variable-width pages, shared across cache entries.
  ///
  /// The cache name folds in the projection (";cols=A,B,C"), so a column read by
  /// three different projections was paged three times and stored three times.
  /// Measured on ClickBench 100M: SearchPhrase (1.2 GB decoded) appeared in three
  /// keys, so 3.6 GB competed for a 2 GB budget and the sweep evicted 25 times --
  /// and an evicted page is an outright lost hit, because a paged column has no
  /// whole-chunk copy to fall back on.
  ///
  /// Keyed by (file identity + filter signature, column, chunk index, chunk rows)
  /// so only genuinely identical row ranges are shared. Safe here because the
  /// coalescer's byte cap is smaller than one row group, so every projection
  /// chunks on row-group boundaries and the cut points coincide -- verified in
  /// the page_directory log, where all unfiltered SearchPhrase entries cut at
  /// exactly 10,000,000 rows regardless of projection.
  ///
  /// Entries keep their own page metadata and take a reference to the buffers, so
  /// usable()/coverage logic is unchanged; only the device allocation is shared.
  /// Guarded by _pinned_entries_mutex.
  std::unordered_map<std::string, shared_variable_pages> _shared_variable_pages;

  /// Distinct values of dictionary-encoded columns, keyed by (file identity +
  /// filter signature, column).
  ///
  /// A decoded STRING column costs its characters plus 4 bytes of offsets per row,
  /// which is why ClickBench's Title (9.22 GB) and URL (8.80 GB) can never be
  /// admitted and the queries that read them get no cache at all. Encoding splits
  /// the column into one int32 code per row -- which is fixed-width, so it needs no
  /// variable-width paging machinery and goes through the existing page path -- plus
  /// one copy of the distinct values, kept here. Measured on ClickBench: Title
  /// 9.22 GB -> 1.58 GB (5.8x), and the keys do not grow as chunks are added
  /// (per-chunk 0.38 GB, unified across chunks 0.38 GB), so this store's size is a
  /// property of the column rather than of how much of it has been cached.
  ///
  /// Held strongly: the codes are meaningless without the keys, so evicting these
  /// while code pages remain would leave unreadable pages behind.
  std::unordered_map<std::string, std::shared_ptr<cudf::column>> _shared_dictionaries;
  std::unordered_set<std::string> _fixed_page_admission_rejected_entries;

  /// Per-query sequencer for opportunistic fadvise calls.  Built fresh
  /// in @ref prepare_for_query, gets one @c pipeline_slot per non-cached
  /// parquet scan (allocated by @ref create_provider_for when it builds
  /// a parquet_split_provider).  The sequencer task is enqueued on the
  /// per-query @c _dispatcher, which injects its own stop_token; the
  /// dispatcher's @c request_stop() in @ref reset() therefore tears the
  /// sequencer down without an extra side-channel.
  std::unique_ptr<load_balancing_scan_batch_coalescer> _metadata_processor;
  io::io_context_registry _ioctx_registry;
};

/// Throw away the page store of whichever scan manager is live, and report the bytes that
/// returned. Declared here rather than reached through the context so the pipeline executor
/// can call it from the OOM retry path without a dependency on SiriusContext. Returns 0 when
/// no scan manager is registered or the store is already empty.
std::size_t drop_page_store_on_oom();

}  // namespace sirius::scan_manager
