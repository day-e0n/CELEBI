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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sirius::scan_manager {

/// Independent of fixed_width_page_state -- kept as its own type (rather than
/// reusing sirius_scan_manager.hpp's) so this header never has to include
/// sirius_scan_manager.hpp, which will need to include *this* header once
/// pinned_entry gains a variable_width_pages_by_column member.
enum class variable_width_page_state : std::uint8_t {
  resident,
  evicted,
};

/// Prototype: a single fixed-*byte-budget* page over a variable-width column.
/// Mirrors fixed_width_column_page's shape, but num_rows is per-page instead
/// of a column-wide constant -- row byte lengths vary per row, so how many
/// rows fit in one page varies too. Kept in a separate page list from
/// fixed_width_column_page (pinned_entry::variable_width_pages_by_column) so
/// the existing fixed-width path is never touched by this one.
struct variable_width_column_page {
  variable_width_column_page() = default;

  variable_width_column_page(variable_width_column_page const& other)
    : table_name(other.table_name),
      column_name(other.column_name),
      state(other.state),
      chunk_index(other.chunk_index),
      page_index(other.page_index),
      start_row(other.start_row),
      num_rows(other.num_rows),
      num_bytes(other.num_bytes),
      memory_space(other.memory_space),
      last_access_tick(other.last_access_tick.load(std::memory_order_relaxed)),
      active_reader_count(other.active_reader_count.load(std::memory_order_relaxed)),
      owned_column(other.owned_column)
  {}

  variable_width_column_page& operator=(variable_width_column_page const& other)
  {
    if (this == &other) { return *this; }
    table_name   = other.table_name;
    column_name  = other.column_name;
    state        = other.state;
    chunk_index  = other.chunk_index;
    page_index   = other.page_index;
    start_row    = other.start_row;
    num_rows     = other.num_rows;
    num_bytes    = other.num_bytes;
    memory_space = other.memory_space;
    last_access_tick.store(other.last_access_tick.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    active_reader_count.store(other.active_reader_count.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    owned_column = other.owned_column;
    return *this;
  }

  variable_width_column_page(variable_width_column_page&& other) noexcept
    : table_name(std::move(other.table_name)),
      column_name(std::move(other.column_name)),
      state(other.state),
      chunk_index(other.chunk_index),
      page_index(other.page_index),
      start_row(other.start_row),
      num_rows(other.num_rows),
      num_bytes(other.num_bytes),
      memory_space(other.memory_space),
      last_access_tick(other.last_access_tick.load(std::memory_order_relaxed)),
      active_reader_count(other.active_reader_count.load(std::memory_order_relaxed)),
      owned_column(std::move(other.owned_column))
  {}

  variable_width_column_page& operator=(variable_width_column_page&& other) noexcept
  {
    if (this == &other) { return *this; }
    table_name   = std::move(other.table_name);
    column_name  = std::move(other.column_name);
    state        = other.state;
    chunk_index  = other.chunk_index;
    page_index   = other.page_index;
    start_row    = other.start_row;
    num_rows     = other.num_rows;
    num_bytes    = other.num_bytes;
    memory_space = other.memory_space;
    last_access_tick.store(other.last_access_tick.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    active_reader_count.store(other.active_reader_count.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
    owned_column = std::move(other.owned_column);
    return *this;
  }

  std::string table_name;
  std::string column_name;
  variable_width_page_state state{variable_width_page_state::resident};
  std::size_t chunk_index{0};
  std::size_t page_index{0};
  /// First row this page covers, relative to the start of its chunk (chunks
  /// are numbered locally, matching fixed_width_column_page::row_offset).
  std::size_t start_row{0};
  std::size_t num_rows{0};
  std::size_t num_bytes{0};
  cucascade::memory::memory_space* memory_space{nullptr};
  mutable std::atomic<std::uint64_t> last_access_tick{0};
  mutable std::atomic<std::uint32_t> active_reader_count{0};
  /// Owned, independently-sliced sub-column (offsets + chars children copied,
  /// not just a view into the original chunk) -- same slice-then-copy-construct
  /// pattern the fixed-width path uses, which is generic cudf::column machinery
  /// and needs no string-specific handling.
  std::shared_ptr<cudf::column> owned_column;
};

/// Sparse row->byte map over one variable-width column chunk -- the "index"
/// that lets page cutting, admission sizing and partial-residency planning all
/// be answered on the host, without touching the GPU or reading a per-row value.
///
/// `byte_prefix[i]` is the number of chars bytes preceding sample row `i`,
/// relative to the column's own first offset. Sample `i` covers row `i * stride`
/// for every `i` except the last, which is pinned to `num_rows` exactly so
/// `byte_prefix.back()` is the column's exact total chars byte size. Between two
/// samples, byte positions are linearly interpolated: the map is therefore exact
/// at stride boundaries and off by at most one stride bucket's byte skew in
/// between -- accurate to well under 1% of a 16MiB page at the default sample
/// budget, while costing a single strided device->host copy (measured 7-18us on
/// a 10M-row chunk, against ~6ms for the full per-row offsets copy an earlier
/// revision paid; see queue_variable_width_offsets_extraction).
struct variable_width_row_byte_map {
  /// Rows between consecutive samples. 0 means the map was never built.
  std::size_t stride{0};
  /// Row count of the chunk this map describes.
  std::size_t num_rows{0};
  /// Cumulative chars bytes at each sample row; see the struct comment for the
  /// sample-row-to-index mapping. Strictly non-decreasing.
  std::vector<std::int64_t> byte_prefix;

  [[nodiscard]] bool valid() const noexcept
  {
    return stride > 0 && num_rows > 0 && byte_prefix.size() >= 2;
  }
  /// Exact total chars bytes of the chunk (0 for an invalid map).
  [[nodiscard]] std::int64_t total_bytes() const noexcept;
  /// Row covered by sample `i` -- `i * stride`, except the final sample which is
  /// `num_rows`. Returns num_rows for any i past the end.
  [[nodiscard]] std::size_t sample_row(std::size_t i) const noexcept;
  /// Interpolated chars bytes preceding `row`. Clamped to [0, num_rows].
  [[nodiscard]] std::int64_t bytes_before(std::size_t row) const noexcept;
  /// Interpolated chars bytes of rows [begin_row, end_row). Never negative.
  [[nodiscard]] std::int64_t bytes_in_range(std::size_t begin_row,
                                            std::size_t end_row) const noexcept;
};

/// Per-row cost the offsets child adds on top of a row's chars bytes. A strings
/// column stores one int32 offset per row (plus a terminating one), so a page's
/// real device footprint is chars_bytes + 4 * rows, not chars_bytes alone.
/// Budgeting without this term is what made narrow string columns degenerate to
/// a single page: o_orderstatus averages 1 chars byte/row, so a 16MiB budget
/// divided by 1 asked for 16M rows per page -- more than any chunk holds -- when
/// the true per-row cost is 5 bytes and the honest answer is ~3.3M rows.
inline constexpr std::size_t kVariableWidthOffsetsOverheadPerRow = sizeof(std::int32_t);

/// One page boundary in the skiplist's TOP level -- the page directory itself.
/// `start_row` is the first row of the page (relative to the chunk) and
/// `start_bytes` the cumulative chars bytes preceding it, so a page's exact row
/// count and byte cost are both differences of adjacent entries. No separate
/// page-start array is kept anywhere: this IS the page list.
struct variable_width_page_boundary {
  std::size_t start_row{0};
  std::int64_t start_bytes{0};
};

/// Multi-level row->byte index over one variable-width column chunk, in array
/// form: level 0 is the densest, each higher level samples the one below by
/// `fanout`, and above them all sits `pages` -- the level that holds only the
/// rows where the running byte total crosses a page budget.
///
/// The node contract the whole structure is built on: key = row, value =
/// cumulative bytes up to that row. Page cutting then needs no separate pass --
/// the top level is produced by walking the levels once and recording every row
/// at which the running total reaches `page_size_bytes`, so the index and the
/// page directory are the same object.
///
/// The levels also mirror where the underlying data can be obtained at all:
///   - Row-granular values (level 0) exist only once the page has been
///     decompressed -- Parquet keeps string lengths inside the compressed,
///     dictionary/RLE-encoded payload, so no amount of footer reading produces
///     them (verified by decompressing a real page and finding the 4-byte
///     length prefixes only after inflation).
///   - Page-granular values exist in the compressed file directly: every
///     Parquet PageHeader carries `num_values` and `uncompressed_page_size`
///     ahead of its payload.
/// So a caller can build the top levels cheaply and only pay for level 0 where
/// exact boundaries are actually needed.
struct variable_width_row_byte_skiplist {
  /// Rows between consecutive level-0 entries. 1 means one entry per row.
  std::size_t base_stride{0};
  /// Sampling ratio between adjacent regular levels.
  std::size_t fanout{0};
  /// Row count of the chunk this index describes.
  std::size_t num_rows{0};
  /// Byte budget the `pages` level was cut for; 0 if it was not cut.
  std::size_t page_size_bytes{0};
  /// Regular-stride levels, level[0] densest. Rows are implicit (index*stride),
  /// so only the cumulative byte value is stored.
  std::vector<std::vector<std::int64_t>> level;
  /// TOP level: the page directory. Irregularly spaced, so rows are explicit.
  /// pages.front().start_row is always 0 when non-empty.
  std::vector<variable_width_page_boundary> pages;

  [[nodiscard]] bool valid() const noexcept
  {
    return base_stride > 0 && fanout > 1 && num_rows > 0 && !level.empty() &&
           level.front().size() >= 2;
  }
  /// Rows between consecutive entries of regular level `lvl`.
  [[nodiscard]] std::size_t stride_of(std::size_t lvl) const noexcept;
  /// Exact total chars bytes of the chunk (0 when invalid).
  [[nodiscard]] std::int64_t total_bytes() const noexcept;
  /// Cumulative chars bytes preceding `row`. Exact when base_stride == 1.
  [[nodiscard]] std::int64_t bytes_before(std::size_t row) const noexcept;
  /// Largest row in (start_row, num_rows] whose span from start_row still fits
  /// `budget` once `per_row_overhead` is charged per row, found by descending
  /// the levels. Never returns start_row, so an oversized single row still
  /// forms a page instead of stalling.
  [[nodiscard]] std::size_t find_page_end(std::size_t start_row, std::int64_t budget,
                                          std::size_t per_row_overhead) const noexcept;
  /// Total host bytes held by all levels plus the page directory.
  [[nodiscard]] std::size_t footprint_bytes() const noexcept;

  // --- the page directory (top level) ---
  [[nodiscard]] std::size_t page_count() const noexcept { return pages.size(); }
  /// Rows in page `i` (the last page runs to num_rows).
  [[nodiscard]] std::size_t page_rows(std::size_t i) const noexcept;
  /// Exact chars bytes of page `i` -- a difference of two stored values.
  [[nodiscard]] std::int64_t page_bytes(std::size_t i) const noexcept;
  /// Index of the page covering `row`, by binary search over the top level.
  /// Returns page_count() if row is out of range.
  [[nodiscard]] std::size_t find_page(std::size_t row) const noexcept;
};

/// Builds the whole structure from a chunk's raw offsets child (host-resident,
/// `num_rows + 1` entries). When `page_size_bytes` is non-zero the top-level
/// page directory is cut in the same pass, charging `per_row_overhead_bytes` per
/// row for the offsets child the page will carry on the GPU.
/// `base_stride == 1` keeps every row and makes page boundaries exact.
variable_width_row_byte_skiplist build_row_byte_skiplist(
  std::vector<std::int32_t> const& offsets, std::size_t base_stride, std::size_t fanout,
  std::size_t page_size_bytes = 0,
  std::size_t per_row_overhead_bytes = kVariableWidthOffsetsOverheadPerRow);

/// Page start rows of `index`, i.e. the top level flattened -- kept as a thin
/// accessor so existing callers that want a plain vector do not have to reach
/// into the structure.
std::vector<std::size_t> cut_pages_with_skiplist(
  variable_width_row_byte_skiplist const& index, std::size_t page_size_bytes,
  std::size_t per_row_overhead_bytes);


/// Cuts [0, map.num_rows) into pages whose real footprint -- chars bytes plus
/// `per_row_overhead_bytes` per row -- stays within `page_size_bytes`, returning
/// each page's first row (always starting at 0, strictly increasing).
///
/// Each boundary is found by binary search over the map's monotone cumulative
/// byte function, so cutting a whole chunk is O(P log N) host arithmetic against
/// a map that already lives in host memory -- no per-row scan, no device round
/// trip. A row whose own cost exceeds the budget still becomes a one-row page
/// rather than stalling the cursor, matching index_variable_width_column_pages.
/// Returns an empty vector for an invalid map or a zero budget.
std::vector<std::size_t> cut_pages_by_byte_budget(variable_width_row_byte_map const& map,
                                                  std::size_t page_size_bytes,
                                                  std::size_t per_row_overhead_bytes);

/// Per-chunk lookup aid: page_start_rows[i] == pages[i].start_row, kept as a
/// parallel array (rather than reading it back out of pages[]) so
/// find_covering_variable_page has a flat, cache-friendly key array to run
/// std::upper_bound over. Strictly increasing by construction, starting at 0
/// -- the variable-width counterpart of fixed_width_chunk_page_span, which
/// stores a single rows_per_page instead because that value is constant there.
struct variable_width_chunk_page_index {
  std::vector<variable_width_column_page> pages;
  std::vector<std::size_t> page_start_rows;
  /// The map the pages were cut from, retained after cutting. It is ~16KiB of
  /// host memory per column-chunk and it is the only thing that can answer
  /// "how many bytes would rows [a, b) cost?" without a device round trip --
  /// which is what admission (sizing a chunk before allocating any of it) and
  /// partial residency (bringing back one evicted page rather than the whole
  /// column) both need. Empty if the pages were cut by the average-based
  /// fallback rather than from a map.
  variable_width_row_byte_map row_byte_map;
  /// Populated instead of row_byte_map on the fixed-slab path; its top level is
  /// this chunk's page directory.
  variable_width_row_byte_skiplist row_byte_skiplist;
};

/// Cuts row_byte_lengths into pages whose total byte size stays
/// <= page_size_bytes wherever possible. A row whose own byte length already
/// exceeds page_size_bytes becomes an oversized page of exactly that one row
/// -- it cannot be split without slicing a value in half, so this is the one
/// case page byte size isn't bounded by page_size_bytes. Pure host arithmetic,
/// no cudf/GPU dependency -- the unit-testable core of the exact-byte-budget
/// approach. Not called by index_variable_width_column_pages_from_view (which
/// cuts by row count from the column's average bytes/row instead, to avoid an
/// O(row count) per-row scan -- see queue_variable_width_offsets_extraction);
/// kept as a tested, correct primitive in case exact per-row budgeting is
/// ever needed again.
variable_width_chunk_page_index index_variable_width_column_pages(
  std::vector<std::size_t> const& row_byte_lengths, std::size_t page_size_bytes);

/// Real entry point: estimates row count per page from the column's exact
/// average bytes/row (see queue_variable_width_offsets_extraction), then
/// materializes an owned sub-column per page via cudf::slice + copy-construct
/// into memory_space's allocator (mirrors fixed_width's
/// materialize_owned_subrange). Returns an empty index if col is not
/// STRING-typed or is empty.
///
/// Convenience wrapper around queue_variable_width_offsets_extraction() +
/// stream.synchronize() + build_variable_width_column_pages() below, for
/// callers indexing a single column. A caller indexing several STRING columns
/// of the same chunk (the common case -- e.g. lineitem has 5) should call
/// those two functions directly instead, to pay for exactly one
/// stream.synchronize() for the whole chunk rather than one per column.
variable_width_chunk_page_index index_variable_width_column_pages_from_view(
  cudf::column_view const& col, std::size_t page_size_bytes, std::size_t chunk_index,
  std::string const& table_name, std::string const& column_name,
  cucascade::memory::memory_space& memory_space, rmm::cuda_stream_view stream);

/// Result of queue_variable_width_offsets_extraction(): the FIRST and LAST
/// offsets of `col`'s row range, host-resident once the stream passed to that
/// call has been synchronized. Their difference is the column's exact total
/// chars byte size, which (divided by row count) gives the exact average
/// bytes/row -- enough to pick a page row-count target without reading a
/// single per-row length. `col` itself is retained (a lightweight view) so
/// build_variable_width_column_pages() can still slice the original column
/// data per page.
struct variable_width_offsets_extraction {
  bool valid{false};
  cudf::column_view col;
  std::int32_t first_offset{0};
  std::int32_t last_offset{0};
  /// Every `sample_stride`'th row offset, host-resident once the stream has
  /// been synchronized -- the raw material for variable_width_row_byte_map.
  /// sampled_offsets[i] is the offset of row `i * sample_stride`; the final
  /// row's offset is last_offset, appended when the map is built. Empty (and
  /// sample_stride 0) if the strided copy could not be issued, in which case
  /// build_variable_width_column_pages falls back to average-based cutting.
  std::vector<std::int32_t> sampled_offsets;
  std::size_t sample_stride{0};
  /// Every row's offset, populated instead of the strided sample when the
  /// caller asks for an exact index (see queue_variable_width_offsets_extraction's
  /// `full_offsets` parameter). Costs a full num_rows*4 byte device->host copy.
  std::vector<std::int32_t> full_offsets;
};

/// Issues a device->host copy of ONLY col's first and last row offsets on
/// `stream` via cudaMemcpyAsync, WITHOUT synchronizing. Returns {valid=false}
/// immediately (no device work issued) if col isn't a non-empty STRING
/// column. O(1) regardless of row count -- earlier revisions here copied
/// every row's offset to compute exact per-row lengths for byte-budget
/// cutting, an O(row count) host-side cost that dominated wall-clock time on
/// SF50 (measured: ~4.3s across a 220-query benchmark run just from this
/// scan, unaffected by page_size_bytes since it's paid regardless of how many
/// pages result). Row-count-based cutting from the exact column average
/// keeps pages close to page_size_bytes -- like fixed-width's own
/// page_size_bytes / element_size division -- without ever reading a
/// per-row value, while a naive fixed-row-count-with-no-byte-awareness
/// scheme would let column-to-column length variance (a short l_shipmode vs.
/// a long l_comment) produce wildly non-uniform page sizes, which is exactly
/// the external-fragmentation risk (in the shared GPU memory pool, not
/// within a page) that byte-aware sizing avoids.
///
/// Callers indexing multiple columns of the same chunk should call this once
/// per column, synchronize `stream` exactly once after all calls, then call
/// build_variable_width_column_pages() for each -- batching what would
/// otherwise be one blocking stream.synchronize() per column into one for the
/// whole chunk.
/// `full_offsets` copies every row offset rather than a strided sample, so the
/// resulting index can be exact. Measured at chunk scale: the copy is ~6ms per
/// 10M rows against ~12us for the sample, and building the skiplist on top costs
/// a further ~47ms -- pay it only when exact page boundaries actually matter.
variable_width_offsets_extraction queue_variable_width_offsets_extraction(
  cudf::column_view const& col, rmm::cuda_stream_view stream, bool full_offsets = false);

/// Assembles the sparse map from an extraction whose stream has already been
/// synchronized. Returns an invalid (default) map -- signalling callers to fall
/// back to average-based cutting -- when the strided sample was never issued or
/// came back non-monotone.
variable_width_row_byte_map build_row_byte_map(
  variable_width_offsets_extraction const& extraction);

/// Must only be called after the stream passed to queue_variable_width_offsets_extraction()
/// for `extraction` has been synchronized (first_offset/last_offset must
/// already be populated). Cuts pages by row count (page_size_bytes divided by
/// the column's exact average bytes/row) and materializes each page's owned
/// sub-column (async device work on `stream`, safely stream-ordered --
/// callers don't need to synchronize again before using the result). Each
/// page's num_bytes is read from the materialized column's own alloc_size()
/// (a host-tracked property once construction completes, no extra device
/// sync) rather than summed from per-row lengths -- real allocated bytes,
/// including the offsets child's overhead, matching the precedent
/// fixed_width_page_resident_alloc_bytes already set for fixed-width pages.
/// Returns an empty index if `extraction.valid` is false.
///
/// `alignment_rows`, when non-zero, overrides byte-budget cutting: pages are cut
/// on that row grid instead (the last one short). This exists because the read
/// path can only avoid a copy when a requested range matches a page *exactly* --
/// materialize_variable_width_chunk_subrange hands back page->owned_column
/// directly in that case, and otherwise slices the partial ends and
/// cudf::concatenate's the range back together, which copies the whole range a
/// second time. Read ranges come from fixed_page_databatch_provider::build_ranges
/// and are one driver fixed-width page each (coalescing defaults to 1), so a page
/// grid equal to that page's row count turns every hit on a paged STRING column
/// into a refcount bump -- strictly cheaper than the whole-chunk fallback, which
/// always deep-copies its slice. Byte-budget cutting is what makes a wide column
/// like l_comment land on ~541K-row pages while the range is 2,097,152 rows, so
/// every single read pays that extra full-range copy.
///
/// The byte budget does not disappear -- it moves to the map, which reports each
/// page's exact byte cost for admission and eviction accounting regardless of
/// where the boundaries were placed.
variable_width_chunk_page_index build_variable_width_column_pages(
  variable_width_offsets_extraction const& extraction, std::size_t page_size_bytes,
  std::size_t chunk_index, std::string const& table_name, std::string const& column_name,
  cucascade::memory::memory_space& memory_space, rmm::cuda_stream_view stream,
  std::size_t alignment_rows = 0, std::size_t slab_max_rows = 0);

/// Picks this column's chars-slab size from a small power-of-two ladder, given
/// how many bytes its rows actually average and how many rows a page may hold.
///
/// One slab size for every column is what made the first slab implementation
/// lose: with a 16MiB chars slab and a 524,288-row cap, l_comment ran at 85%
/// utilization but l_returnflag at 14% and l_shipmode at 23%, because the row cap
/// binds long before the byte budget on narrow columns. Measured consequence at a
/// 2GiB budget on SF30: 26 eviction events against 7, 67 cache hits against 77,
/// and 5.1% slower end to end -- the internal waste cost far more than the ~6.7%
/// external fragmentation the slabs removed. Sizing per column keeps utilization
/// high while still drawing from only a handful of distinct sizes, which measured
/// 0.16-0.81% external fragmentation against 6.7% for fully variable sizes.
///
/// The ladder is capped below by `min_bytes` so tiny columns still get a sane
/// slab, and above by `max_bytes`.
std::size_t choose_chars_slab_bytes(double bytes_per_row, std::size_t max_rows_per_page,
                                    std::size_t min_bytes, std::size_t max_bytes);

/// Materializes rows [start_row, start_row + num_rows) of `col` into page storage
/// whose device allocations are a FIXED size, independent of how much of them the
/// page actually uses.
///
/// This is the point of the whole exercise. Measured on this box with
/// cudaMallocAsync (the allocator behind memory_space), churning a working set
/// with 50% eviction for 20 rounds: allocations that are all exactly one size
/// cost 0.00% external fragmentation at every scale tested, while variable-sized
/// allocations cost 3.9-13.0% (6.7% at a 6GB working set). Rounding sizes up to
/// 2MB/4MB/power-of-two classes did NOT help -- only true uniformity did. A cudf
/// STRING column is two allocations (a chars buffer and an offsets child), so
/// both are given their own fixed size here; the unused tail of each is internal
/// fragmentation, which costs capacity but never strands a slot.
///
/// `chars_slab_bytes` must be >= the page's chars byte count and
/// `offsets_slab_bytes` >= (num_rows + 1) * sizeof(int32_t); the caller sizes
/// pages so that holds. Returns nullptr if the range does not fit, if `col` is
/// not a non-empty STRING column, or if it carries nulls (the prototype leaves
/// nullable columns on the exact-size path rather than hand-rolling a sliced
/// null mask).
std::shared_ptr<cudf::column> materialize_page_into_slab(
  cudf::column_view const& col, std::size_t start_row, std::size_t num_rows,
  std::size_t chars_slab_bytes, std::size_t offsets_slab_bytes,
  cucascade::memory::memory_space& memory_space, rmm::cuda_stream_view stream);

/// Fills `index.pages` -- the top level, i.e. the page directory -- in place, using
/// the slab constraints (chars must fit `chars_budget_bytes`, row count must not
/// exceed `max_rows_per_page`).
///
/// This exists as a second step because slab sizing is circular: the chars slab
/// class is chosen from the column's measured average width, which is only known
/// once the levels below exist. Building the levels first and finalizing the top
/// level afterwards keeps the page directory inside the structure rather than in a
/// parallel array beside it.
void finalize_page_directory(variable_width_row_byte_skiplist& index,
                             std::size_t chars_budget_bytes,
                             std::size_t max_rows_per_page);

/// Page boundaries for one variable-width column chunk, found by a SINGLE batched
/// binary search on the GPU -- no host copy of the offsets, no auxiliary index.
///
/// The offsets child of a cudf STRING column is already a prefix sum over row byte
/// lengths, and it is already sorted and already resident. So "how many rows fit in
/// the next `chars_budget_bytes`" is not a scan and not a running total: it is
/// `lower_bound(offsets, offsets[0] + k * budget)` for k = 1, 2, 3, ... , and every
/// k can be searched at once. That is the whole difference from fixed-width, where
/// the same question is a division because bytes-per-row is a schema constant.
///
/// Because the search targets sit on an even byte grid, a page can overshoot the
/// budget by at most one row's bytes (a row is never split), which is the same
/// bound the row-by-row cut gives. `max_rows_per_page` additionally caps the row
/// count so the page's offsets buffer stays a fixed size.
///
/// Returns page start rows (always beginning at 0, strictly increasing), or an
/// empty vector if `col` is not a non-empty STRING column with int32 offsets.
std::vector<std::size_t> find_page_starts_on_device(cudf::column_view const& col,
                                                    std::size_t chars_budget_bytes,
                                                    std::size_t max_rows_per_page,
                                                    rmm::cuda_stream_view stream);

/// Cuts pages for the fixed-slab layout: chars must fit `chars_budget_bytes` and
/// row count must not exceed `max_rows_per_page` (so the offsets slab is also
/// bounded). Unlike cut_pages_by_byte_budget this charges no per-row overhead
/// against the chars budget -- the offsets live in their own slab and are capped
/// by the row limit instead.
std::vector<std::size_t> cut_pages_for_slabs(variable_width_row_byte_skiplist const& index,
                                             std::size_t chars_budget_bytes,
                                             std::size_t max_rows_per_page);

/// Binary search over idx.page_start_rows (O(log P), replacing the O(P) linear
/// scan this stands in for) for the page covering `row`, a row offset
/// relative to the same chunk index was built from. Returns nullptr if row is
/// out of range for this chunk (empty index, or row >= chunk's row count) or
/// the covering page has been evicted.
variable_width_column_page const* find_covering_variable_page(
  variable_width_chunk_page_index const& idx, std::size_t row);

/// True while at least one in-flight scan holds a reference to this page --
/// mirrors fixed_width_page_is_active; eviction must skip these.
bool variable_width_page_is_active(variable_width_column_page const& page);

/// Bumps/drops page's reader refcount (delta = +1 acquire / -1 release).
void adjust_variable_width_page_active_reader(variable_width_column_page const& page, int delta);

/// Marks page as most-recently-used for LRU eviction ordering.
void touch_variable_width_page(variable_width_column_page const& page);

/// Single shared LRU tick source for BOTH fixed-width and variable-width
/// pages. apply_global_page_cache_memory_pressure (sirius_scan_manager.cpp)
/// sorts eviction candidates of both page types together by last_access_tick
/// to find the globally coldest ones -- that comparison is only meaningful if
/// both types draw ticks from the same monotonic counter. Exposed here (this
/// header is already included by sirius_scan_manager.cpp) so
/// touch_fixed_width_page can call it too, instead of each page type
/// incrementing its own independent counter from zero -- which would make a
/// variable-width page's tick incomparable to a fixed-width page's (fixed-width
/// pages get touched far more often, e.g. every lineitem chunk insert, so its
/// counter races ahead; sorting the two together by raw value would then
/// always look like every variable-width page is colder than every
/// fixed-width one, regardless of true recency).
std::uint64_t next_shared_page_lru_tick();

}  // namespace sirius::scan_manager
