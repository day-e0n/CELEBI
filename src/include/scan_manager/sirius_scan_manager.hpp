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
class cache_entry_info {
 public:
  std::vector<std::string> resolved_file_paths;    ///< parquet identity (file set)
  std::string catalog_name;                        ///< duckdb identity: catalog (attach alias)
  std::string schema_name;                         ///< duckdb identity: schema
  std::string table_name;                          ///< duckdb identity: table
  std::string filter_signature;                    ///< non-empty for filter-specific auto caches
  duckdb::vector<duckdb::ColumnIndex> column_ids;  ///< cached columns, by primary index
  std::vector<std::string> names;                  ///< aligned with column_ids; gather keys

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

/**
 * @brief A single pinned-table entry, keyed by table name in the scan_manager.
 *
 * Stores the column projection captured at pin time (so the scan side knows
 * which columns the user pinned) along with the data batches making up the
 * pinned table. The vector may be empty until splits are populated.
 */
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
  /// Target page size used when fixed_width_pages_by_column was built.
  std::size_t fixed_width_page_size_bytes{0};
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
  [[nodiscard]] std::size_t insert_fixed_page_entry_from_view(
    const std::string& name,
    cache_entry_info cache_info,
    cudf::table_view view,
    cucascade::memory::memory_space& memory_space,
    rmm::cuda_stream_view stream);

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

 private:
  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void start_metadata_processing();

  /// \brief Attach a cached batch_provider to @p op if a pinned entry can serve
  ///        it. Returns true when a cache hit was assigned (the caller then skips
  ///        the disk-reading split_provider for this operator).
  bool try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op);

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

}  // namespace sirius::scan_manager
