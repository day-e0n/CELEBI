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
#include <sstream>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sirius::scan_manager {

namespace {
/// The live scan manager, so drop_page_store_on_oom() can reach its page store without the
/// pipeline executor knowing about SiriusContext. There is one scan manager per context and
/// one context per process here; a second one simply replaces the first.
std::atomic<sirius_scan_manager*> g_page_store_owner{nullptr};
}  // namespace


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

/// Whether a scan carrying dynamic filters may read from the page cache.
///
/// Reading a cached entry here is safe regardless of the filter: DYNAMIC_FILTER is a
/// separate operator sitting above the scan, so it masks cached batches exactly as it
/// masks freshly decoded ones. Cached batches are also handed out as
/// filter_state::UNFILTERED, so nothing downstream assumes the rows were pre-pruned.
///
/// The two directions are NOT symmetric, so they get separate variables.
///
/// Write-only is strictly worse than leaving both shut: an SF50 run with only the write
/// gate open cached lineitem and orders pages that no dynamic-filter scan could ever read
/// back, and the VRAM they took cut cache hits from 88 to 37 and cost 8.8% of scan time.
/// Opening both is worse still on SF100 -- caching a probe-side lineitem scan means
/// caching every row the dynamic filter would have discarded, which exceeded the budget
/// and failed with "GPU pipeline task exceeded maximum OOM retry limit (100)" at the same
/// query in 3 of 3 runs.
///
/// Read-only is the direction those two results leave open, and it is the cheap one: a
/// dynamic-filter scan reads entries that ordinary unfiltered scans of the same file
/// already paid for, so it adds hits without adding a single resident byte. Enabling the
/// read gate therefore does not imply the write gate, and `..._SCANS=1` still opens both
/// for the older configuration.
bool dynamic_filter_scan_cache_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_DYNAMIC_FILTER_SCANS");
  return value != nullptr && std::string_view(value) == "1";
}

bool dynamic_filter_scan_cache_read_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_DYNAMIC_FILTER_REUSE");
  if (value != nullptr && std::string_view(value) == "1") { return true; }
  return dynamic_filter_scan_cache_enabled();
}

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

/// Prototype variable-width (STRING) page-level caching, entirely separate
/// from the fixed-width path -- off by default so fixed-width behavior is
/// unaffected either way. When on, variable_width_pages_by_column is built
/// alongside (not instead of) data_batches_by_column's existing whole-chunk
/// copy, so the hybrid-provider fallback keeps working unchanged if this
/// path finds no page-level hit.
bool variable_width_page_cache_enabled()
{
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_CACHE_ENABLED");
  return value != nullptr && std::string_view(value) == "1";
}

/// Defined further down alongside the other fixed-width page knobs; needed here
/// by index_variable_width_columns_for_chunk to derive the read-range row grid.
std::size_t fixed_width_page_size_bytes();

/// Build the variable-width page index from every row's offset (exact page
/// boundaries) instead of a 2048-entry strided sample. Measured at chunk scale:
/// exact costs ~53ms per 10M rows against ~12us for the sample, and produced the
/// same page count -- so this is off by default and exists to make that
/// trade-off measurable rather than argued.
bool variable_width_page_exact_offsets()
{
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_EXACT_OFFSETS");
  return value != nullptr && std::string_view(value) == "1";
}

/// Give every variable-width page identically-sized device allocations (a fixed
/// chars slab plus a fixed offsets slab) instead of exact-fit ones. Measured with
/// cudaMallocAsync, 50%-eviction churn for 20 rounds: uniform allocations cost
/// 0.00% external fragmentation at every working-set size tested, exact-fit ones
/// 3.9-13.0% (6.7% at 6GB). Rounding to 2MB/4MB/power-of-two classes did not help
/// -- only true uniformity did. The unused tail of each slab is internal
/// fragmentation: it costs capacity but can never strand a slot.
/// ON by default: uniform allocations are the reason variable-width paging exists.
/// Set SIRIUS_VARIABLE_PAGE_SLABS=0 to fall back to exact-fit pages, which reintroduces
/// the external fragmentation this is here to remove.
bool variable_width_page_slabs_enabled()
{
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_SLABS");
  return value == nullptr || std::string_view(value) != "0";
}

/// Rows a slab page may hold, which fixes the offsets slab at
/// (this + 1) * 4 bytes. Default 524288 -> a 2MiB offsets slab, paired with the
/// 16MiB chars slab from variable_width_page_size_bytes().
std::size_t variable_width_page_slab_max_rows()
{
  static constexpr std::size_t kDefault = 524288;
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_SLAB_MAX_ROWS");
  if (value == nullptr || value[0] == '\0') { return kDefault; }
  try {
    auto parsed = static_cast<std::size_t>(std::stoull(value));
    return parsed == 0 ? kDefault : parsed;
  } catch (...) {
    return kDefault;
  }
}

/// Cut variable-width pages on the read-range row grid instead of a byte budget,
/// so a cache hit can hand back a page by reference rather than concatenating a
/// range back together. On by default; set to 0 to A/B against byte-budget
/// cutting. See build_variable_width_column_pages' header comment.
bool variable_width_page_align_to_read_range()
{
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_ALIGN_TO_READ_RANGE");
  return value == nullptr || std::string_view(value) != "0";
}

std::size_t variable_width_page_size_bytes()
{
  static constexpr std::size_t kDefaultPageBytes = 16ULL * 1024ULL * 1024ULL;
  auto const* value = std::getenv("SIRIUS_VARIABLE_WIDTH_PAGE_BYTES");
  if (value == nullptr || value[0] == '\0') { return kDefaultPageBytes; }
  try {
    auto parsed = static_cast<std::size_t>(std::stoull(value));
    return parsed == 0 ? kDefaultPageBytes : parsed;
  } catch (...) {
    return kDefaultPageBytes;
  }
}

/// Builds this chunk's page index for one STRING column and records it on entry --
/// a no-op unless variable_width_page_cache_enabled(). Callers use the return value to
/// skip the whole-chunk copy in data_batches_by_column when paging succeeded (the paged
/// copy is then authoritative and cheaper to keep than a redundant second copy). Shared
/// by both insert_fixed_page_entry_from_view call sites (append into an existing entry,
/// and create a brand-new one).
/// Per-(entry, column) cap on cached bytes -- default 512MiB. Without this, a
/// single scan touching many chunks of a huge table (lineitem at SF50: one
/// TEST run reproduced a real OOM crash caching l_returnflag/l_linestatus
/// chunk-by-chunk across a 300M-row scan, faster than reactive eviction
/// could keep up within that same still-running query) can grow one column's
/// resident footprint without bound. Once over the cap, new chunks for that
/// column are simply left unindexed (page cache stays whatever it already
/// had) rather than growing further -- asymmetric on purpose: cheap to check,
/// no need to unwind what's already resident and possibly in use.
std::size_t variable_width_page_admission_max_column_bytes()
{
  static constexpr std::size_t kDefault = 512ULL * 1024ULL * 1024ULL;
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_ADMISSION_MAX_COLUMN_BYTES");
  if (value == nullptr || value[0] == '\0') { return kDefault; }
  try {
    auto parsed = static_cast<std::size_t>(std::stoull(value));
    return parsed == 0 ? kDefault : parsed;
  } catch (...) {
    return kDefault;
  }
}

/// Upper bound on a single chunk's row count for it to be eligible for
/// variable-width paging at all -- default 12M rows. Session-measured on
/// SF50 TPC-H: fixed-width page caching only ever gets REUSED (a later
/// query's try_assign_cached_entries actually finding a usable, resident
/// entry -- confirmed via production logs, "assigned pinned entry" never
/// once fired for lineitem or orders across a full 10x benchmark run) for
/// the small dimension tables -- customer (~7.5M rows), supplier (~375K),
/// part (~10M), nation (25), region (5) at SF50 -- which scan in chunks at
/// or under this size. lineitem (~300M rows) and orders (~75M rows) chunks
/// run tens of millions of rows and their entries are always either
/// admission-rejected or evicted (by the ~21 OTHER queries' entries
/// competing for the same shared budget) before a later query can reuse
/// them, so paging their STRING columns (l_comment, o_comment, ...) is pure
/// write-side cost for a read that essentially never happens. Gating on
/// chunk size steers variable-width caching toward exactly the tables where
/// it can pay off, mirroring where fixed-width paging itself already pays
/// off, instead of applying it uniformly regardless of table scale.
std::size_t variable_width_page_max_chunk_rows()
{
  static constexpr std::size_t kDefault = 12'000'000ULL;
  auto const* value = std::getenv("SIRIUS_VARIABLE_PAGE_MAX_CHUNK_ROWS");
  if (value == nullptr || value[0] == '\0') { return kDefault; }
  try {
    auto parsed = static_cast<std::size_t>(std::stoull(value));
    return parsed == 0 ? kDefault : parsed;
  } catch (...) {
    return kDefault;
  }
}

std::size_t variable_width_column_resident_bytes(pinned_entry const& entry,
                                                  std::string const& column_name)
{
  auto it = entry.variable_width_pages_by_column.find(column_name);
  if (it == entry.variable_width_pages_by_column.end()) { return 0; }
  std::size_t total = 0;
  for (auto const& chunk_idx : it->second) {
    for (auto const& page : chunk_idx.pages) {
      if (page.state == variable_width_page_state::resident) { total += page.num_bytes; }
    }
  }
  return total;
}

/// Batches STRING-column page indexing for every non-fixed-width column of one
/// chunk behind a SINGLE stream.synchronize(), instead of one sync per column.
/// index_variable_width_column_pages_from_view's device->host offsets read
/// needs a stream barrier to make the copy host-visible; paying that barrier
/// once per column (as an earlier version of this code did) meant e.g. 5
/// blocking synchronizes per chunk for lineitem's 5 STRING columns. Queuing
/// every column's async copy first and synchronizing once cuts that to 1.
///
/// Returns the set of column indices (into column_names/fixed_columns) that
/// were fully paged -- callers use this to skip the whole-chunk copy in
/// data_batches_by_column for those columns, since a fully-paged chunk is
/// already served by the read path's variable_width_page_covers()/
/// materialize_variable_width_chunk_subrange() checks (chunk_column_covers/
/// materialize_chunk_subrange try those first). A column absent from the
/// returned set was left unindexed (disabled, non-STRING, admission-capped,
/// or empty column) -- callers must fall back to the whole-chunk copy for it.
/// Identity of a column-chunk for the shared page store: everything that makes
/// two scans' buffers interchangeable, and nothing about which projection asked.
/// The file set and filter signature fix the row set; chunk index and row count
/// fix the range within it.
/// Whether the entry holds the whole table. A short entry -- left behind by a
/// mid-scan auto_cache_populate_failed, or (once partial residency lands) by
/// design -- would otherwise serve fewer rows than the table has, with nothing
/// anywhere comparing num_rows against the truth. Unknown total (non-parquet
/// reader) is treated as complete, preserving today's behaviour.
/// Whether this exact set of row groups is already in the entry -- the duplicate
/// test that replaces the old row-count heuristic. Empty provenance (non-parquet
/// reader) can never match, so those callers fall back to appending, which is the
/// behaviour they had before provenance existed.
/// Index of the chunk holding exactly these row groups, or npos.
///
/// With one entry per projection, chunk_index was just an arrival counter and that
/// was harmless -- every column of an entry arrived together in the same insert.
/// Merging projections breaks that: a later insert brings different columns for
/// row groups already present, and appending them at a fresh index would make the
/// same chunk_index mean different rows for different columns, which
/// all_columns_cover_tiled would then assemble into one batch. Wrong results, not
/// a miss. Looking the chunk up by its row groups is what keeps a chunk index
/// meaning one fixed range of the table.
std::size_t find_chunk_by_provenance(pinned_entry const& entry,
                                     chunk_provenance const& incoming)
{
  if (incoming.slices.empty()) { return std::numeric_limits<std::size_t>::max(); }
  for (std::size_t i = 0; i < entry.chunk_provenance_by_index.size(); ++i) {
    if (entry.chunk_provenance_by_index[i].slices == incoming.slices) { return i; }
  }
  return std::numeric_limits<std::size_t>::max();
}

bool chunk_already_present(pinned_entry const& entry, chunk_provenance const& incoming)
{
  if (incoming.slices.empty()) { return false; }
  for (auto const& existing : entry.chunk_provenance_by_index) {
    if (existing.slices == incoming.slices) { return true; }
  }
  return false;
}

/// Whether the entry holds every row group of every file it names. Used for
/// filtered entries, whose row count cannot be compared against the table total.
/// Unknown provenance (non-parquet reader) reads as complete, preserving the
/// behaviour those callers had before provenance existed.
bool entry_covers_all_row_groups(pinned_entry const& entry)
{
  auto const& expected = entry.cache_info.expected_row_groups;
  if (expected.empty() || entry.chunk_provenance_by_index.empty()) { return true; }
  std::unordered_map<std::string, std::unordered_set<int>> held;
  for (auto const& chunk : entry.chunk_provenance_by_index) {
    for (auto const& [path, groups] : chunk.slices) {
      for (auto const rg : groups) { held[path].insert(static_cast<int>(rg)); }
    }
  }
  for (auto const& [path, want] : expected) {
    auto it = held.find(path);
    if (it == held.end()) { return false; }
    for (auto const rg : want) {
      if (!it->second.contains(rg)) { return false; }
    }
  }
  return true;
}

/// Whether the cache may serve a scan only partly, leaving the rest to parquet.
/// Off until the residual read path lands -- see make_provider_for_pinned_entry.
/// Whether an oversized STRING column is cached as dictionary codes instead of
/// being refused. Off by default until measured.
/// Whether one cache entry per (file, filter) holds the union of every projection's
/// columns. Off until measured -- it changes cache identity, so every entry built
/// under one setting is unusable under the other.
/// Whether a STRING column may be cached as a whole chunk when variable-width
/// paging is off. On by default (existing behaviour); set to 0 to get a genuinely
/// fixed-width-only cache, which is the right control for measuring what caching
/// strings is worth at all.
bool string_columns_cacheable()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_CACHE_STRING_COLUMNS");
  return value == nullptr || std::string_view(value) != "0";
}

bool column_keyed_cache_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_COLUMN_KEYED_CACHE");
  return value != nullptr && std::string_view(value) == "1";
}

bool dictionary_encoding_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_DICTIONARY_ENCODE");
  return value != nullptr && std::string_view(value) == "1";
}

/// Identity of a dictionary: the row set it describes, and the column. Chunk index
/// is deliberately absent -- one key set serves every chunk of the column, which is
/// what makes the store's size independent of how much has been cached.
std::string shared_dictionary_key(pinned_entry const& entry, std::string const& column_name)
{
  std::ostringstream out;
  for (auto const& path : entry.cache_info.resolved_file_paths) { out << path << '\x1f'; }
  out << '\x1e' << entry.cache_info.filter_signature << '\x1e' << column_name;
  return out.str();
}

bool partial_residency_enabled()
{
  auto const* value = std::getenv("SIRIUS_FIXED_PAGE_PARTIAL_RESIDENCY");
  return value != nullptr && std::string_view(value) == "1";
}

/// Whether the entry holds the whole table, ignoring the escape hatch.
///
/// Kept separate from entry_is_complete because the residual decision needs the
/// FACT, not the policy: with the hatch open, entry_is_complete answers "serve it
/// anyway", and a caller that read that as "it holds everything" would skip naming
/// the complement and hand the scan a fraction of the table.
bool entry_holds_whole_table(pinned_entry const& entry)
{
  // Only an UNFILTERED entry can be checked against the table's row count. A
  // filtered scan legitimately holds fewer rows -- `#4 != ''` keeps 13,172,392 of
  // 99,997,497 -- and comparing those against the footer total rejected 137 of
  // them per run, which is what cut celebi_fixed from -31.5% to -14.5% while this
  // check was written wrong.
  //
  // Completeness for a filtered entry is a row-GROUP question, not a row-count
  // one: it is complete when it holds every row group the scan read. That is what
  // chunk_provenance records, so use it and fall back to the row count only for
  // the unfiltered case where the footer total is directly comparable.
  // Row groups are the right unit for BOTH cases -- a filtered entry's row count
  // is legitimately smaller than the table's, so comparing it against the footer
  // total rejected 137 entries per run and cut celebi_fixed from -31.5% to -14.5%
  // while this check was written that way. The row-count comparison survives only
  // as a fallback for readers that supply no provenance.
  if (!entry.cache_info.expected_row_groups.empty()) {
    return entry_covers_all_row_groups(entry);
  }
  auto const total = entry.cache_info.table_total_rows;
  return total == 0 || entry.num_rows == total;
}

bool entry_is_complete(pinned_entry const& entry)
{
  // Escape hatch to reproduce the pre-check behaviour for an A/B. Set to 0 and a
  // half-populated entry is served again, which is only safe now that the caller
  // names the row groups it did NOT get and the reader fetches them: num_rows
  // accumulates as chunks arrive, so an entry is legitimately incomplete while it
  // is being filled, and usable()'s `_covered_rows == _entry.num_rows` is
  // satisfied at every intermediate state.
  auto const* relax = std::getenv("SIRIUS_FIXED_PAGE_REQUIRE_COMPLETE_ENTRY");
  if (relax != nullptr && std::string_view(relax) == "0") { return true; }
  return entry_holds_whole_table(entry);
}

std::string shared_variable_page_key(pinned_entry const& entry,
                                     std::string const& column_name,
                                     std::size_t chunk_index,
                                     std::size_t num_rows)
{
  // Identity must be the ROW GROUPS, not chunk_index: that index is an arrival
  // counter, so the same row group lands at a different index in every entry and
  // a chunk_index-keyed store never hits. Measured: pages_shared fired 0 times
  // until this was keyed by provenance.
  std::ostringstream out;
  out << entry.cache_info.filter_signature << '\x1e' << column_name << '\x1e' << num_rows;
  if (chunk_index < entry.chunk_provenance_by_index.size()) {
    for (auto const& [path, groups] : entry.chunk_provenance_by_index[chunk_index].slices) {
      out << '\x1e' << path;
      for (auto const rg : groups) { out << '\x1f' << rg; }
    }
    return out.str();
  }
  // No provenance (non-parquet reader): fall back to an identity that can only
  // match within one entry, which is to say it never shares. Correct, not useful.
  out << '\x1e' << entry.cache_info.filter_signature << '\x1e' << chunk_index;
  for (auto const& path : entry.cache_info.resolved_file_paths) { out << '\x1f' << path; }
  return out.str();
}

/// Encode one chunk of a STRING column into int32 codes against a key set shared by
/// every chunk of that column, returning the codes (or null if it cannot be done).
///
/// The keys have to be shared or the codes are meaningless across chunks:
/// cudf::dictionary::encode builds its own key set per call, so code 1 would name a
/// different string in every chunk. The first chunk establishes the keys; later
/// chunks are re-mapped onto them with set_keys, which also adds any values they
/// introduce. Measured on ClickBench, chunks overlap almost completely (Title's keys
/// were 0.38 GB per chunk and 0.38 GB unified), so this converges rather than
/// growing with the number of chunks.
///
/// Returns null when re-mapping would drop values -- set_keys nulls out rows whose
/// value is not in the target key set, which would silently corrupt the column, so
/// the caller falls back to storing the chunk as strings.
std::shared_ptr<cudf::column> encode_column_as_dictionary(
  pinned_entry& entry,
  std::string const& name,
  std::string const& column_name,
  cudf::column_view const& column_view,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream,
  std::unordered_map<std::string, std::shared_ptr<cudf::column>>* store)
{
  if (store == nullptr) { return nullptr; }
  try {
    auto const key = shared_dictionary_key(entry, column_name);
    auto existing  = store->find(key);

    auto encoded = cudf::dictionary::encode(
      column_view, cudf::data_type{cudf::type_id::INT32}, stream,
      memory_space.get_default_allocator());

    if (existing == store->end()) {
      cudf::dictionary_column_view dcv(encoded->view());
      auto keys = std::make_shared<cudf::column>(
        dcv.keys(), stream, memory_space.get_default_allocator());
      auto codes = std::make_shared<cudf::column>(
        dcv.get_indices_annotated(), stream, memory_space.get_default_allocator());
      (*store)[key] = std::move(keys);
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] dictionary_encoded table='{}' column='{}' rows={} keys={}",
        name, column_name, column_view.size(), dcv.keys_size());
      return codes;
    }

    // Re-map onto the established keys, adding whatever this chunk introduces.
    auto merged = cudf::dictionary::add_keys(
      cudf::dictionary_column_view(encoded->view()), existing->second->view(), stream,
      memory_space.get_default_allocator());
    cudf::dictionary_column_view merged_view(merged->view());
    if (merged_view.null_count() > column_view.null_count()) { return nullptr; }

    auto keys = std::make_shared<cudf::column>(
      merged_view.keys(), stream, memory_space.get_default_allocator());
    auto codes = std::make_shared<cudf::column>(
      merged_view.get_indices_annotated(), stream, memory_space.get_default_allocator());
    existing->second = std::move(keys);
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] dictionary_remapped table='{}' column='{}' rows={} keys={}",
      name, column_name, column_view.size(), merged_view.keys_size());
    return codes;
  } catch (std::exception const& e) {
    SIRIUS_LOG_INFO("[fixed-page-cache] dictionary_encode_failed table='{}' column='{}' what='{}'",
                    name, column_name, e.what());
    return nullptr;
  }
}

std::unordered_set<std::size_t> index_variable_width_columns_for_chunk(
  pinned_entry& entry,
  std::string const& name,
  std::vector<std::string> const& column_names,
  std::vector<bool> const& fixed_columns,
  cudf::table_view const& view,
  std::size_t chunk_index,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream,
  std::vector<std::size_t> const& cache_info_projected_bytes,
  std::unordered_map<std::string, shared_variable_pages>* shared_pages)
{
  std::unordered_set<std::size_t> paged;
  if (!variable_width_page_cache_enabled()) { return paged; }
  if (static_cast<std::size_t>(view.num_rows()) > variable_width_page_max_chunk_rows()) {
    return paged;
  }

  bool const slabs         = variable_width_page_slabs_enabled();
  // Fixed-size paging no longer needs the offsets on the host: the boundary
  // search runs on the device against the offsets child in place.
  bool const exact_offsets = variable_width_page_exact_offsets();
  auto const slab_max_rows = slabs ? variable_width_page_slab_max_rows() : 0;
  std::vector<std::pair<std::size_t, variable_width_offsets_extraction>> pending;
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    if (fixed_columns[i]) { continue; }
    auto const column_view = view.column(static_cast<cudf::size_type>(i));
    if (column_view.type().id() != cudf::type_id::STRING || column_view.size() <= 0) { continue; }
    // Per-column admission, decided ONCE from the parquet footer rather than
    // incrementally from what is already resident.
    //
    // The old test was `resident_bytes(col) >= 512MB`, which cuts a column in
    // half: chunks before the threshold are paged (their whole-chunk slot left
    // nullptr) and chunks after it are stored as whole chunks. A column in that
    // mixed state can never be served by the chunk fallback --
    // has_chunk_backing_for_selected_columns requires EVERY chunk non-null --
    // and which chunks land on which side depends on materialize task arrival
    // order. Measured on ClickBench 100M: that order-dependence was the entire
    // run-to-run variance of the combined condition (24/40/48 hits, sd 42x the
    // single-cache conditions); pinning the cap so no column is cut made it
    // deterministic at 34/34.
    //
    // Deciding from the whole-table projected size instead means a column is
    // either paged everywhere or nowhere. A refused column keeps the
    // whole-chunk path it has in the fixed-width-only configuration, which is
    // what makes it survivable: paged columns have no second copy, so evicting
    // one costs a hit outright (measured 1:1 -- disabling variable eviction
    // recovered exactly the 12 lost hits and 12 coverage failures).
    auto const projected = (cache_info_projected_bytes.size() == column_names.size())
                             ? cache_info_projected_bytes[i]
                             : std::size_t{0};
    if (projected != 0 && projected > variable_width_page_admission_max_column_bytes()) {
      SIRIUS_LOG_INFO(
        "[variable-page-cache] admission_skip reason=projected_column_bytes table='{}' "
        "column='{}' projected_bytes={} max_column_bytes={}",
        name,
        column_names[i],
        projected,
        variable_width_page_admission_max_column_bytes());
      continue;
    }
    if (projected == 0 && variable_width_column_resident_bytes(entry, column_names[i]) >=
                            variable_width_page_admission_max_column_bytes()) {
      continue;  // no footer estimate (non-parquet reader): fall back to the old test
    }
    auto extraction =
      queue_variable_width_offsets_extraction(column_view, stream, exact_offsets);
    if (!extraction.valid) { continue; }
    pending.emplace_back(i, std::move(extraction));
  }
  if (pending.empty()) { return paged; }

  // Row grid the read path will ask for: fixed_page_databatch_provider builds one
  // range per driver fixed-width page (coalescing defaults to 1) and picks the
  // driver with the fewest rows per page -- i.e. the widest element type. Cutting
  // variable pages on that same grid makes range and page coincide, which is the
  // only case materialize_variable_width_chunk_subrange can skip its copy.
  std::size_t alignment_rows = 0;
  if (!exact_offsets && !slabs && variable_width_page_align_to_read_range()) {
    std::size_t widest_element = 0;
    for (std::size_t i = 0; i < column_names.size(); ++i) {
      if (!fixed_columns[i]) { continue; }
      auto const type = view.column(static_cast<cudf::size_type>(i)).type();
      if (!cudf::is_fixed_width(type)) { continue; }
      widest_element = std::max<std::size_t>(widest_element, cudf::size_of(type));
    }
    // No fixed-width column in this chunk means no grid to align to (e.g. the
    // SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0 experiment); fall back to bytes.
    if (widest_element > 0) {
      alignment_rows = std::max<std::size_t>(1, fixed_width_page_size_bytes() / widest_element);
    }
  }

  stream.synchronize();

  if (entry.variable_width_page_size_bytes == 0) {
    entry.variable_width_page_size_bytes = variable_width_page_size_bytes();
  }
  for (auto& [i, extraction] : pending) {
    // Adopt an identical column-chunk another projection already paged, instead of
    // cutting and copying a second set of buffers for the same rows. The copy is
    // of page METADATA only; every page's owned_column is a shared_ptr, so the
    // device allocation is shared rather than duplicated.
    auto const shared_key = shared_variable_page_key(
      entry, column_names[i], chunk_index, static_cast<std::size_t>(extraction.col.size()));
    bool adopted = false;
    variable_width_chunk_page_index idx;
    if (shared_pages != nullptr) {
      auto shared_it = shared_pages->find(shared_key);
      if (shared_it != shared_pages->end() &&
          shared_it->second.buffers.size() == shared_it->second.index.pages.size() &&
          !shared_it->second.buffers.empty()) {
        // All-or-nothing: a partially-freed row cannot serve a chunk, and taking
        // the live half would produce an index with holes that usable() would
        // reject anyway.
        std::vector<std::shared_ptr<cudf::column>> locked;
        locked.reserve(shared_it->second.buffers.size());
        for (auto const& weak : shared_it->second.buffers) {
          auto strong = weak.lock();
          if (!strong) { break; }
          locked.push_back(std::move(strong));
        }
        if (locked.size() == shared_it->second.buffers.size()) {
          idx = shared_it->second.index;
          for (std::size_t p = 0; p < idx.pages.size(); ++p) {
            idx.pages[p].owned_column = locked[p];
            idx.pages[p].state        = variable_width_page_state::resident;
            idx.pages[p].memory_space = &memory_space;
          }
          adopted = true;
          SIRIUS_LOG_INFO(
            "[variable-page-cache] pages_shared table='{}' column='{}' chunk_index={} pages={}",
            name,
            column_names[i],
            chunk_index,
            idx.pages.size());
        } else {
          shared_pages->erase(shared_it);  // buffers gone; stop offering this row
        }
      }
    }
    if (!adopted) {
      idx = build_variable_width_column_pages(extraction,
                                              entry.variable_width_page_size_bytes,
                                              chunk_index,
                                              name,
                                              column_names[i],
                                              memory_space,
                                              stream,
                                              alignment_rows,
                                              slab_max_rows);
      if (!idx.pages.empty() && shared_pages != nullptr) {
        shared_variable_pages record;
        record.index = idx;
        record.buffers.reserve(idx.pages.size());
        for (auto& page : record.index.pages) {
          record.buffers.emplace_back(page.owned_column);
          page.owned_column.reset();  // the store must not own the allocation
        }
        (*shared_pages)[shared_key] = std::move(record);
      }
    }
    if (idx.pages.empty()) { continue; }
    auto const num_pages   = idx.pages.size();
    auto const column_size = extraction.col.size();
    auto& chunks_idx        = entry.variable_width_pages_by_column[column_names[i]];
    if (chunks_idx.size() <= chunk_index) { chunks_idx.resize(chunk_index + 1); }
    chunks_idx[chunk_index] = std::move(idx);
    paged.insert(i);
    SIRIUS_LOG_INFO(
      "[variable-page-cache] page_directory indexed table='{}' column='{}' chunk_index={} "
      "pages={} page_bytes={} rows={} align_rows={}",
      name,
      column_names[i],
      chunk_index,
      num_pages,
      entry.variable_width_page_size_bytes,
      column_size,
      alignment_rows);
  }
  return paged;
}

/// Whether column_name is fully served by the variable-width page index across
/// every chunk of entry -- i.e. this column can be materialized purely from
/// variable_width_pages_by_column, with no whole-chunk data_batches_by_column
/// fallback needed (insert_fixed_page_entry_from_view deliberately leaves that
/// fallback null for a successfully-paged column, to avoid the redundant
/// double-copy). index_variable_width_columns_for_chunk either pages a
/// column's entire chunk or leaves it unpaged (no partial-chunk paging), so a
/// non-empty page list for every chunk implies full coverage.
///
/// Without this check, choose_driver_column() and
/// has_chunk_backing_for_selected_columns() only recognize data_batches_by_column
/// as valid backing -- so any entry containing a paged column looked entirely
/// unusable to BOTH fixed_page_databatch_provider and cached_databatch_provider,
/// and make_provider_for_pinned_entry fell through to nullptr, forcing a full
/// cache-bypass rescan from disk on every hit instead of the fast page-cache
/// read path (materialize_chunk_subrange already handles paged columns
/// correctly -- it was only the availability check blocking entry to it).
bool variable_width_column_fully_paged(pinned_entry const& entry, std::string const& column_name)
{
  auto it = entry.variable_width_pages_by_column.find(column_name);
  if (it == entry.variable_width_pages_by_column.end() || it->second.empty()) { return false; }
  for (auto const& chunk_idx : it->second) {
    if (chunk_idx.pages.empty()) { return false; }
  }
  return true;
}

/// Whether entry's variable-width page index (if any) fully covers
/// [row_offset, row_offset+num_rows) of column_name's chunk_index'th chunk --
/// the variable-width read-side counterpart of fixed-width's
/// column_pages_cover_tiled, called from chunk_column_covers/
/// materialize_chunk_subrange to prefer the fine-grained page cache over the
/// existing whole-chunk fallback whenever it can serve the request.
bool variable_width_page_covers(pinned_entry const& entry,
                                std::string const& column_name,
                                std::size_t chunk_index,
                                std::size_t row_offset,
                                std::size_t num_rows)
{
  auto it = entry.variable_width_pages_by_column.find(column_name);
  if (it == entry.variable_width_pages_by_column.end() || chunk_index >= it->second.size()) {
    return false;
  }
  auto const& chunk_idx = it->second[chunk_index];
  auto const range_end  = row_offset + num_rows;
  auto cursor           = row_offset;
  while (cursor < range_end) {
    auto const* page = find_covering_variable_page(chunk_idx, cursor);
    if (!page || !page->owned_column) { return false; }
    auto const page_end = page->start_row + page->num_rows;
    if (page_end <= cursor) { return false; }
    cursor = std::min(page_end, range_end);
  }
  return true;
}

/// Materializes [row_offset, row_offset+num_rows) of column_name's
/// chunk_index'th chunk by walking variable-width pages (touching each for
/// LRU) and concatenating pieces if the range spans more than one -- the
/// variable-width counterpart of materialize_page_tiled_column. Returns
/// nullptr if the range isn't fully page-covered (caller falls back to the
/// existing whole-chunk materialize_chunk_subrange in that case).
std::shared_ptr<cudf::column> materialize_variable_width_chunk_subrange(
  pinned_entry const& entry,
  std::string const& column_name,
  std::size_t chunk_index,
  std::size_t row_offset,
  std::size_t num_rows,
  cucascade::memory::memory_space& memory_space)
{
  auto it = entry.variable_width_pages_by_column.find(column_name);
  if (it == entry.variable_width_pages_by_column.end() || chunk_index >= it->second.size()) {
    return nullptr;
  }
  auto const& chunk_idx = it->second[chunk_index];

  auto const range_end = row_offset + num_rows;
  auto cursor          = row_offset;
  std::vector<std::shared_ptr<cudf::column>> pieces;
  std::vector<cudf::column_view> piece_views;
  while (cursor < range_end) {
    auto const* page = find_covering_variable_page(chunk_idx, cursor);
    if (!page || !page->owned_column) { return nullptr; }
    touch_variable_width_page(*page);
    auto const page_end  = page->start_row + page->num_rows;
    auto const piece_end = std::min(page_end, range_end);
    if (piece_end <= cursor) { return nullptr; }

    std::shared_ptr<cudf::column> piece;
    if (cursor == page->start_row && piece_end == page_end) {
      piece = page->owned_column;  // whole page matches -- reuse, no extra copy
    } else {
      auto const begin = static_cast<cudf::size_type>(cursor - page->start_row);
      auto const end    = static_cast<cudf::size_type>(begin + (piece_end - cursor));
      auto views        = cudf::slice(page->owned_column->view(), {begin, end});
      if (views.empty()) { return nullptr; }
      piece = std::make_shared<cudf::column>(
        views.front(), cudf::get_default_stream(), memory_space.get_default_allocator());
    }
    piece_views.emplace_back(piece->view());
    pieces.push_back(std::move(piece));
    cursor = piece_end;
  }
  if (pieces.empty()) { return nullptr; }
  if (pieces.size() == 1) { return pieces.front(); }
  auto concatenated = cudf::concatenate(
    piece_views, cudf::get_default_stream(), memory_space.get_default_allocator());
  return std::shared_ptr<cudf::column>{std::move(concatenated)};
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

void touch_fixed_width_page(fixed_width_column_page const& page)
{
  page.last_access_tick.store(next_shared_page_lru_tick(), std::memory_order_relaxed);
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

  // O(1) path: within a chunk, pages are laid out contiguously in row order with a
  // constant rows_per_page (the last page may be a shorter remainder), so the page
  // covering row_offset is `row_offset / rows_per_page` -- no scan needed. The
  // subsequent fixed_page_covers_range check is retained as-is: it still verifies
  // residency and the exact [row_offset, row_offset+num_rows) bound (rejecting a
  // range that spans into a following page, or a stale/evicted page), so behavior
  // is identical to the previous linear scan, just located in O(1) instead of O(P).
  auto spans_it = entry.fixed_width_chunk_page_spans.find(column_name);
  if (spans_it != entry.fixed_width_chunk_page_spans.end() &&
      chunk_index < spans_it->second.size()) {
    auto const& span = spans_it->second[chunk_index];
    if (span.page_count > 0) {
      auto const first = span.page_start_index;
      auto const last  = first + span.page_count;
      if (last <= pages_it->second.size()) {
        // rows_per_page == 0 marks a chunk whose pages were cut on row-group
        // boundaries, so the stride is not constant and division cannot locate a
        // page. Pages within a span stay sorted by row_offset either way, so fall
        // back to a binary search -- O(log P) over the few dozen pages a chunk
        // holds, against the O(1) the uniform case keeps.
        std::size_t index = 0;
        if (span.rows_per_page > 0) {
          index = row_offset / span.rows_per_page;
          if (index >= span.page_count) { return nullptr; }
          index += first;
        } else {
          auto const begin = pages_it->second.begin() + static_cast<std::ptrdiff_t>(first);
          auto const end   = pages_it->second.begin() + static_cast<std::ptrdiff_t>(last);
          auto it          = std::upper_bound(
            begin, end, row_offset, [](std::size_t offset, fixed_width_column_page const& page) {
              return offset < page.row_offset;
            });
          if (it == begin) { return nullptr; }
          index = static_cast<std::size_t>(std::distance(pages_it->second.begin(), it) - 1);
        }
        auto const& page = pages_it->second[index];
        if (fixed_page_covers_range(page, chunk_index, row_offset, num_rows)) { return &page; }
        return nullptr;
      }
    }
  }
  return nullptr;
}

bool pinned_entry_has_active_reader(pinned_entry const& entry)
{
  for (auto const& [_, pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : pages) {
      if (fixed_width_page_is_active(page)) { return true; }
    }
  }
  return false;
}

/// Drop `it` from `entries` when replacing it with an incompatible re-insert
/// (schema/filter mismatch under the same cache name). fixed_page_databatch_provider
/// and cached_databatch_provider hold a raw `pinned_entry const&` into this map,
/// grabbed under _pinned_entries_mutex at construction time but read again later
/// (e.g. the destructor's adjust_active_pages(-1)) with no lock held and no
/// shared ownership of the entry itself. A plain erase() here would free that
/// map node out from under any provider still using it -- a real use-after-free
/// that can surface much later as unrelated-looking heap corruption. If any page
/// still has an active reader, keep the node alive by re-keying it into the map
/// under a throwaway name instead of erasing it outright; it becomes unreachable
/// to future lookups (which only ever look up the canonical name) but its
/// storage stays valid until the referencing provider(s) are done with it.
void orphan_or_erase_pinned_entry(std::unordered_map<std::string, pinned_entry>& entries,
                                  std::unordered_map<std::string, pinned_entry>::iterator it)
{
  if (!pinned_entry_has_active_reader(it->second)) {
    entries.erase(it);
    return;
  }
  static std::atomic<std::uint64_t> orphan_id{0};
  auto node = entries.extract(it);
  node.key() = node.key() + "::orphan#" + std::to_string(orphan_id.fetch_add(1));
  entries.insert(std::move(node));
}

class fixed_page_databatch_provider final : public databatch_provider {
 public:
  explicit fixed_page_databatch_provider(
    pinned_entry const& entry,
    std::span<size_t> selected_columns,
    std::unordered_map<std::string, std::shared_ptr<cudf::column>> const* dictionaries = nullptr,
    std::unordered_map<std::string, std::unordered_set<int>> const* allowed_row_groups = nullptr,
    std::vector<cache_filter_range> const* consumer_filter_ranges                      = nullptr,
    bool consumer_filter_analyzable                                                    = false,
    std::string consumer_filter_signature                                              = {})
    : _entry(entry), _dictionaries(dictionaries), _allowed_row_groups(allowed_row_groups)
  {
    if (consumer_filter_ranges != nullptr) {
      _consumer_filter_checked    = true;
      _consumer_filter_ranges     = *consumer_filter_ranges;
      _consumer_filter_analyzable = consumer_filter_analyzable;
      _consumer_filter_signature  = std::move(consumer_filter_signature);
    }
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

  /// Whether this provider can serve anything at all.
  ///
  /// This used to require `_covered_rows == _entry.num_rows` -- the whole entry or
  /// nothing. That is why evicting ONE page cost an entire entry, including its
  /// columns that were still fully resident, and why variable-width eviction had
  /// to be entry-granular ("freeing only part of it would strand the rest").
  ///
  /// Now the provider serves the ROW GROUPS it fully covers and leaves the rest to
  /// the scan, which reads them from parquet as it would on a miss. The row group
  /// is the unit because that is the unit the reader's residual path excludes
  /// (set_cached_row_groups) and the unit pages are cut on; a chunk holds several,
  /// so making the chunk the unit threw away every fully resident row group that
  /// happened to share a chunk with a missing one.
  [[nodiscard]] bool usable() const noexcept { return !_ranges.empty(); }

  /// Whether every chunk the entry holds is served from cache, i.e. the scan has
  /// no residual to read. Callers that cannot handle a residual check this.
  [[nodiscard]] bool covers_entire_entry() const noexcept
  {
    return !_ranges.empty() && _covered_rows == _entry.num_rows;
  }

  /// Chunk indices this provider serves at least part of -- for logging only.
  /// The authoritative set is covered_row_groups().
  [[nodiscard]] std::vector<std::size_t> const& covered_chunks() const noexcept
  {
    return _covered_chunks;
  }

  /// Row groups this provider serves in full, per file. The scan excludes exactly
  /// these and reads what is left.
  [[nodiscard]] std::unordered_map<std::string, std::unordered_set<int>> const&
  covered_row_groups() const noexcept
  {
    return _covered_row_groups;
  }

  [[nodiscard]] std::size_t batch_count() const noexcept { return _ranges.size(); }

  /// Total physical 16MB fixed-width pages served by this hit, across all
  /// selected columns -- range.page_count is the coalesced page run along the
  /// driver column, and all_columns_cover_tiled (build_ranges) guarantees every
  /// selected column shares the same page-aligned row boundaries for that run.
  [[nodiscard]] std::size_t total_page_count() const noexcept
  {
    std::size_t pages = 0;
    for (auto const& range : _ranges) { pages += range.page_count; }
    return pages * _column_names.size();
  }

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
      if (!chunk_filter_allows(page.chunk_index)) { continue; }
      if (!all_columns_cover_tiled(page.chunk_index, page.row_offset, page.num_rows)) { continue; }

      std::size_t coalesced_rows  = page.num_rows;
      std::size_t coalesced_pages = 1;
      auto const range_begin      = page.row_offset;
      // Never coalesce across a row-group boundary. Coverage is decided per row
      // group below, and a range that straddled two of them could not be assigned
      // to either without splitting it back apart.
      auto const& layout            = layout_for(page.chunk_index);
      auto const coalesce_row_limit = [&] {
        if (layout.empty()) { return std::numeric_limits<std::size_t>::max(); }
        auto const ord = row_group_ordinal(layout, range_begin);
        return ord < layout.size() ? layout[ord].row_end : std::numeric_limits<std::size_t>::max();
      }();
      while (coalesced_pages < max_coalesce_pages && i + coalesced_pages < driver_pages.size()) {
        auto const& next = driver_pages[i + coalesced_pages];
        if (next.state != fixed_width_page_state::resident || next.num_rows == 0) { break; }
        if (next.chunk_index != page.chunk_index || next.memory_space != page.memory_space) { break; }
        if (next.row_offset != range_begin + coalesced_rows) { break; }
        auto const next_rows = coalesced_rows + next.num_rows;
        if (range_begin + next_rows > coalesce_row_limit) { break; }
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

    drop_partially_covered_row_groups();
  }

  /// One row group's extent inside a chunk, in the chunk's own row coordinates.
  struct chunk_row_group {
    std::string const* file_path{nullptr};
    int row_group{0};
    std::size_t row_begin{0};
    std::size_t row_end{0};
  };

  /// Row groups a chunk holds, in the order their rows appear in the chunk.
  ///
  /// Empty when the chunk's provenance cannot name them -- a non-parquet producer
  /// records no per-row-group row counts, and a count list that disagrees with the
  /// chunk's own row total is not trustworthy enough to cut on. Both cases fall
  /// back to chunk-granular coverage, which claims less and never claims wrong.
  [[nodiscard]] std::vector<chunk_row_group> chunk_row_groups(std::size_t chunk_index) const
  {
    std::vector<chunk_row_group> layout;
    if (chunk_index >= _entry.chunk_provenance_by_index.size()) { return layout; }
    auto const& provenance = _entry.chunk_provenance_by_index[chunk_index];
    if (provenance.row_group_rows.empty()) { return layout; }
    std::size_t row   = 0;
    std::size_t index = 0;
    for (auto const& [path, groups] : provenance.slices) {
      for (auto const group : groups) {
        if (index >= provenance.row_group_rows.size()) { return {}; }
        auto const rows = provenance.row_group_rows[index++];
        layout.push_back({&path, static_cast<int>(group), row, row + rows});
        row += rows;
      }
    }
    if (index != provenance.row_group_rows.size() || row != provenance.num_rows) { return {}; }
    return layout;
  }

  [[nodiscard]] std::vector<chunk_row_group> const& layout_for(std::size_t chunk_index) const
  {
    auto it = _row_group_layouts.find(chunk_index);
    if (it == _row_group_layouts.end()) {
      it = _row_group_layouts.emplace(chunk_index, chunk_row_groups(chunk_index)).first;
    }
    return it->second;
  }

  /// Ordinal of the row group containing @p row_offset, or layout.size() if none.
  [[nodiscard]] static std::size_t row_group_ordinal(std::vector<chunk_row_group> const& layout,
                                                     std::size_t row_offset)
  {
    for (std::size_t ordinal = 0; ordinal < layout.size(); ++ordinal) {
      if (row_offset >= layout[ordinal].row_begin && row_offset < layout[ordinal].row_end) {
        return ordinal;
      }
    }
    return layout.size();
  }

  /// Whether this chunk's rows include every row the consuming scan wants.
  ///
  /// A chunk holds {r : P(r)} for the predicate P that produced it. The scan wants
  /// {r : Q(r)}. Serving is safe exactly when Q implies P, which for conjunctions
  /// of range comparisons is a containment test per column. With the predicate out
  /// of the cache key, this is what keeps a chunk from being handed to a query
  /// whose rows it does not all hold.
  [[nodiscard]] bool chunk_filter_allows(std::size_t chunk_index) const
  {
    if (!_consumer_filter_checked) { return true; }
    if (chunk_index >= _entry.chunk_provenance_by_index.size()) { return false; }
    auto const& provenance = _entry.chunk_provenance_by_index[chunk_index];
    if (provenance.filter_analyzable && provenance.filter_ranges.empty()) { return true; }
    // Identical predicate: reusable whether or not it can be read as ranges. This
    // is what the filter key used to do, and range containment ADDS to it rather
    // than replacing it -- most ClickBench predicates have no range form at all.
    if (!provenance.filter_signature.empty() &&
        provenance.filter_signature == _consumer_filter_signature) {
      return true;
    }
    return filter_ranges_subsume(provenance.filter_ranges,
                                 provenance.filter_analyzable,
                                 _consumer_filter_ranges,
                                 _consumer_filter_analyzable);
  }

  /// Whether this one row group survives the scan's predicate.
  [[nodiscard]] bool row_group_allowed(std::string const& file_path, int row_group) const
  {
    if (_allowed_row_groups == nullptr) { return true; }
    auto it = _allowed_row_groups->find(file_path);
    if (it == _allowed_row_groups->end()) { return false; }
    return it->second.contains(row_group);
  }

  /// Whether every row group this chunk holds survives the scan's predicate.
  ///
  /// A chunk is all-or-nothing (its row groups are read together), so one pruned
  /// row group disqualifies it and the reader takes the whole chunk -- where the
  /// stats filter then skips exactly that row group. Serving it from cache
  /// instead would deliver rows the pruning exists to avoid, which is the cost
  /// that made opening the dynamic-filter read gate a 10.2% regression.
  [[nodiscard]] bool chunk_row_groups_allowed(std::size_t chunk_index) const
  {
    if (_allowed_row_groups == nullptr) { return true; }
    if (chunk_index >= _entry.chunk_provenance_by_index.size()) { return true; }
    for (auto const& [path, groups] : _entry.chunk_provenance_by_index[chunk_index].slices) {
      auto it = _allowed_row_groups->find(path);
      if (it == _allowed_row_groups->end()) { return false; }
      for (auto const rg : groups) {
        if (!it->second.contains(static_cast<int>(rg))) { return false; }
      }
    }
    return true;
  }

  /// Keep the row groups the ranges tile completely; discard the rest.
  ///
  /// A half-covered ROW GROUP cannot be handed out -- the missing rows would simply
  /// be absent from the scan's output, silently. A half-covered CHUNK can be, because
  /// the reader excludes row groups rather than chunks (set_cached_row_groups), so
  /// the cached and residual halves stay exactly complementary at this granularity
  /// too. Deciding per chunk instead threw away every fully resident row group that
  /// shared a chunk with a missing one: on ClickBench's reorder run that refusal
  /// fired 54 times at 8,388,608-row row groups against 6 at 10,000,000-row ones,
  /// and cost 12.5s of the 25.7s best case.
  void drop_partially_covered_row_groups()
  {
    if (_ranges.empty()) { return; }

    for (auto const& range : _ranges) {
      if (layout_for(range.chunk_index).empty()) {
        // Provenance cannot name this chunk's row groups; fall back to all-or-nothing.
        drop_partially_covered_chunks();
        return;
      }
    }

    std::unordered_map<std::size_t, std::vector<std::size_t>> covered_rows;
    for (auto const& range : _ranges) {
      auto const& layout = layout_for(range.chunk_index);
      auto const ordinal = row_group_ordinal(layout, range.row_offset);
      if (ordinal == layout.size()) { continue; }
      // A range that crosses a row-group boundary belongs wholly to neither, and
      // counting it against the one it starts in would credit that row group with
      // rows it does not hold -- enough to read as complete and then serve short.
      // build_ranges stops coalescing at boundaries, so this only fires when a
      // single PAGE straddles one (pages are cut on boundaries only when the
      // producer recorded row-group row counts). Leave it uncounted: the row group
      // then reads as incomplete and the scan fetches it from parquet.
      if (range.row_offset + range.num_rows > layout[ordinal].row_end) { continue; }
      auto& rows = covered_rows.try_emplace(range.chunk_index, layout.size(), 0).first->second;
      rows[ordinal] += range.num_rows;
    }

    // Chunk order is fixed so the choice below is reproducible run to run.
    std::vector<std::size_t> chunks;
    chunks.reserve(covered_rows.size());
    for (auto const& [chunk_index, rows] : covered_rows) { chunks.push_back(chunk_index); }
    std::sort(chunks.begin(), chunks.end());

    std::unordered_map<std::size_t, std::vector<bool>> served;
    _covered_row_groups.clear();
    for (auto const chunk_index : chunks) {
      auto const& rows   = covered_rows.at(chunk_index);
      auto const& layout = layout_for(chunk_index);
      std::vector<bool> keep(layout.size(), false);
      for (std::size_t ordinal = 0; ordinal < layout.size(); ++ordinal) {
        auto const expected = layout[ordinal].row_end - layout[ordinal].row_begin;
        if (expected == 0 || rows[ordinal] != expected) { continue; }
        if (!row_group_allowed(*layout[ordinal].file_path, layout[ordinal].row_group)) { continue; }
        // Serve each row group from exactly one chunk. A merged entry can hold the
        // same row group under two chunk indices; serving both would hand the scan
        // that row group twice, and the reader -- which excludes a row group once,
        // by name -- cannot undo the second copy. Seen as counts ABOVE the correct
        // maximum, not below it.
        auto& claimed = _covered_row_groups[*layout[ordinal].file_path];
        if (!claimed.insert(layout[ordinal].row_group).second) { continue; }
        keep[ordinal] = true;
      }
      served.emplace(chunk_index, std::move(keep));
    }

    std::vector<fixed_page_batch_range> kept;
    kept.reserve(_ranges.size());
    _covered_rows = 0;
    std::unordered_set<std::size_t> touched_chunks;
    for (auto const& range : _ranges) {
      auto const& layout = layout_for(range.chunk_index);
      auto const ordinal = row_group_ordinal(layout, range.row_offset);
      auto const it      = served.find(range.chunk_index);
      if (ordinal == layout.size() || it == served.end() || !it->second[ordinal]) { continue; }
      if (range.row_offset + range.num_rows > layout[ordinal].row_end) { continue; }
      _covered_rows += range.num_rows;
      touched_chunks.insert(range.chunk_index);
      kept.push_back(range);
    }
    _ranges = std::move(kept);
    _covered_chunks.assign(touched_chunks.begin(), touched_chunks.end());
    std::sort(_covered_chunks.begin(), _covered_chunks.end());
  }

  /// Keep only chunks whose ranges tile the chunk completely; discard the rest.
  ///
  /// A half-covered chunk cannot be handed out: the missing rows would simply be
  /// absent from the scan's output, silently. Dropping it sends the whole chunk to
  /// the disk path, which costs a re-read of rows that happen to be resident but
  /// keeps the cached-plus-residual split exactly complementary -- the invariant
  /// the whole partial-residency design rests on.
  void drop_partially_covered_chunks()
  {
    if (_ranges.empty()) { return; }
    std::unordered_map<std::size_t, std::size_t> rows_by_chunk;
    for (auto const& range : _ranges) { rows_by_chunk[range.chunk_index] += range.num_rows; }

    std::unordered_set<std::size_t> complete;
    for (auto const& [chunk_index, rows] : rows_by_chunk) {
      if (!chunk_row_groups_allowed(chunk_index)) { continue; }
      auto const expected = chunk_index < _entry.chunk_provenance_by_index.size()
                              ? _entry.chunk_provenance_by_index[chunk_index].num_rows
                              : 0;
      // No provenance (non-parquet reader): keep today's behaviour and trust the
      // ranges, since there is no residual path for those callers anyway.
      if (expected == 0 || rows == expected) { complete.insert(chunk_index); }
    }

    std::vector<fixed_page_batch_range> kept;
    kept.reserve(_ranges.size());
    _covered_rows = 0;
    for (auto const& range : _ranges) {
      if (!complete.contains(range.chunk_index)) { continue; }
      _covered_rows += range.num_rows;
      kept.push_back(range);
    }
    _ranges = std::move(kept);
    _covered_chunks.assign(complete.begin(), complete.end());
    std::sort(_covered_chunks.begin(), _covered_chunks.end());

    _covered_row_groups.clear();
    for (auto const chunk_index : _covered_chunks) {
      if (chunk_index >= _entry.chunk_provenance_by_index.size()) { continue; }
      for (auto const& [path, groups] : _entry.chunk_provenance_by_index[chunk_index].slices) {
        for (auto const group : groups) {
          _covered_row_groups[path].insert(static_cast<int>(group));
        }
      }
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
        if (hybrid && (has_chunk_column(_column_names[i]) ||
                       variable_width_column_fully_paged(_entry, _column_names[i]))) {
          continue;
        }
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
    if (variable_width_page_covers(_entry, column_name, chunk_index, row_offset, num_rows)) {
      return true;
    }
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
    if (auto paged = materialize_variable_width_chunk_subrange(_entry,
                                                              column_name,
                                                              range.chunk_index,
                                                              range.row_offset,
                                                              range.num_rows,
                                                              *range.memory_space)) {
      return paged;
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
        // Dictionary-encoded columns are stored as int32 codes. Consumers expect the
        // STRING column the scan would have produced from disk, so rebuild it here:
        // the batch is a couple of million rows, not the whole table, so this
        // materializes far less than the column the cache would otherwise have had
        // to hold in full.
        if (_entry.dictionary_encoded_columns.contains(column_name)) {
          return decode_dictionary_column(column_name, std::move(owned));
        }
        return owned;
      }
    }

    if (auto tiled = materialize_page_tiled_column(column_name, range)) { return tiled; }

    if (!fixed_page_hybrid_provider_enabled()) { return nullptr; }
    return materialize_chunk_subrange(column_name, range);
  }

  /// Rebuild the STRING column a batch's consumers expect from its int32 codes.
  /// Returns null when the keys are missing, which makes the provider fall through
  /// to the disk path rather than hand back codes as if they were strings.
  [[nodiscard]] std::shared_ptr<cudf::column> decode_dictionary_column(
    std::string const& column_name, std::shared_ptr<cudf::column> codes) const
  {
    if (!_dictionaries) { return nullptr; }
    auto it = _dictionaries->find(shared_dictionary_key(_entry, column_name));
    if (it == _dictionaries->end() || !it->second) { return nullptr; }
    try {
      auto dict = cudf::make_dictionary_column(
        std::make_unique<cudf::column>(it->second->view()),
        std::make_unique<cudf::column>(codes->view()));
      return std::shared_ptr<cudf::column>{
        cudf::dictionary::decode(cudf::dictionary_column_view(dict->view()))};
    } catch (std::exception const& e) {
      SIRIUS_LOG_INFO("[fixed-page-cache] dictionary_decode_failed column='{}' what='{}'",
                      column_name, e.what());
      return nullptr;
    }
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
  std::vector<std::size_t> _covered_chunks;
  /// Set when the cache key carries no predicate, so each chunk must be cleared
  /// against this scan's own predicate instead.
  bool _consumer_filter_checked{false};
  bool _consumer_filter_analyzable{false};
  std::vector<cache_filter_range> _consumer_filter_ranges;
  std::string _consumer_filter_signature;
  /// Row groups served in full, per file -- what the scan excludes from its read.
  std::unordered_map<std::string, std::unordered_set<int>> _covered_row_groups;
  /// Memoized chunk_row_groups() results; keyed by chunk index.
  mutable std::unordered_map<std::size_t, std::vector<chunk_row_group>> _row_group_layouts;
  pinned_entry const& _entry;
  std::unordered_map<std::string, std::shared_ptr<cudf::column>> const* _dictionaries{nullptr};
  /// Row groups the scan's predicate cannot rule out, or null when the caller
  /// could not compute them. Chunks outside it are left to the reader, which
  /// skips them by stats -- serving them from cache would hand back rows the
  /// pruning exists to avoid.
  std::unordered_map<std::string, std::unordered_set<int>> const* _allowed_row_groups{nullptr};
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
  pinned_entry const& entry,
  std::span<size_t> selected_columns,
  std::unordered_map<std::string, std::shared_ptr<cudf::column>> const* dictionaries,
  std::unordered_map<std::string, std::unordered_set<int>> const* allowed_row_groups,
  std::vector<cache_filter_range> const* consumer_filter_ranges,
  bool consumer_filter_analyzable,
  std::string consumer_filter_signature)
{
  if (fixed_page_backed_provider_enabled()) {
    auto fixed_page_provider =
      std::make_unique<fixed_page_databatch_provider>(entry,
                                                      selected_columns,
                                                      dictionaries,
                                                      allowed_row_groups,
                                                      consumer_filter_ranges,
                                                      consumer_filter_analyzable,
                                                      std::move(consumer_filter_signature));
    // Partial residency is only safe once the scan reads the complement. Until the
    // residual path exists (row-group exclusion in the ingestible, and a worker
    // loop that runs the cached and provider sources in sequence rather than as an
    // either/or), a partial provider would emit its chunks and silently drop the
    // rest. So the whole-entry requirement stays in force by default; the env var
    // enables the partial path for the A/B that measures it.
    bool const serviceable = partial_residency_enabled()
                               ? fixed_page_provider->usable()
                               : fixed_page_provider->covers_entire_entry();
    if (serviceable) {
      SIRIUS_LOG_INFO("[fixed-page-cache] using {} cached provider batches={} pages={}",
                      fixed_page_hybrid_provider_enabled() ? "hybrid page/chunk" : "page-backed",
                      fixed_page_provider->batch_count(),
                      fixed_page_provider->total_page_count());
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
    // With dictionary encoding on, a STRING column enters the entry as one int32
    // code per row -- the characters go to the shared key store, whose size is a
    // property of the column's distinct values rather than of the rows cached
    // here. Charging the string estimate instead rejected exactly the columns the
    // encoding exists to admit: measured on ClickBench, the four entries carrying
    // URL/Title were refused at identical incoming_bytes with the encoding on and
    // off, so no oversized string column ever reached the encoder.
    if (dictionary_encoding_enabled() && col.type().id() == cudf::type_id::STRING) {
      bytes += static_cast<std::size_t>(col.size()) * sizeof(std::int32_t);
      continue;
    }
    // Conservative admission estimate for variable-width hybrid chunks.
    bytes += static_cast<std::size_t>(col.size()) * 32ULL;
  }
  return bytes;
}

/// Bytes this entry holds as FIXED-WIDTH PAGES -- what the fixed-width admission
/// limit is meant to bound.
std::size_t fixed_width_entry_logical_bytes(pinned_entry const& entry)
{
  std::size_t bytes = 0;
  for (auto const& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto const& page : col_pages) {
      bytes += page.num_bytes;
    }
  }
  return bytes;
}

/// Bytes this entry holds as WHOLE-CHUNK copies -- the columns that could not be
/// paged, which on ClickBench means STRING columns like URL and Title. These have
/// their own budget and must be counted separately from the pages: summing the two
/// into the fixed check let 7.47 GB of chunks fill a 6 GB fixed limit while only
/// 440 MB of pages existed, so a single column (UserID) was paged and every
/// widening after it was refused.
std::size_t entry_whole_chunk_bytes(pinned_entry const& entry)
{
  std::size_t bytes = 0;
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
  // Experiment-only escape hatch: SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0
  // forces every column through the non-page (hybrid/whole-chunk) path, so a
  // run can isolate variable_width_page_cache_enabled()'s effect with
  // fixed-width paging held OFF instead of both always being on together.
  // Unset (the default) leaves this function's original behavior untouched.
  auto const* fixed_enabled = std::getenv("SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED");
  if (fixed_enabled != nullptr && std::string_view(fixed_enabled) == "0") { return false; }
  switch (col.type().id()) {
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::EMPTY: return false;
    default: return true;
  }
}

/// is_fixed_width_page_candidate() without the SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0
/// override -- this column's real type, regardless of that experiment flag.
/// Guards the whole-chunk fallback store below: without this check, setting
/// that flag routed every numeric/date column through the *unbounded*
/// whole-chunk cache meant for the rare string/list column (no admission cap,
/// no eviction -- that path was never built to hold an entire table), which
/// reproduced a real OOM (GPU pipeline task exceeded its retry limit, then
/// crashed again during error-path cleanup) on a single large lineitem scan.
/// Skipping the whole-chunk store for a column that's only "not fixed-width"
/// because of the flag -- as opposed to genuinely variable-width -- leaves it
/// with no caching at all under that flag, matching baseline for that column.
bool is_intrinsically_fixed_width_type(cudf::column_view const& col) noexcept
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
  auto const page_start_index = pages.size();

  // Row-group boundaries inside this chunk, as offsets from its start. A page
  // must not straddle one: the residual path reads whole row groups, so a page
  // spanning two cannot be dropped or kept as a unit, and the row-group resize
  // that removes remainder waste only works if pages restart at each boundary.
  // Empty when the producer supplied no row counts (non-parquet reader), which
  // falls back to the plain byte grid this used to be.
  std::vector<std::size_t> boundaries;
  if (chunk_index < entry.chunk_provenance_by_index.size()) {
    std::size_t acc = 0;
    for (auto const rg_rows : entry.chunk_provenance_by_index[chunk_index].row_group_rows) {
      acc += rg_rows;
      if (acc >= rows) { break; }
      boundaries.push_back(acc);
    }
  }
  auto next_boundary = [&](std::size_t from) {
    auto it = std::upper_bound(boundaries.begin(), boundaries.end(), from);
    return it == boundaries.end() ? rows : *it;
  };

  for (std::size_t row_offset = 0; row_offset < rows;) {
    auto const page_rows =
      std::min({rows_per_page, rows - row_offset, next_boundary(row_offset) - row_offset});
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
    page.last_access_tick.store(next_shared_page_lru_tick(), std::memory_order_relaxed);
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
    row_offset += page_rows;
  }

  auto const page_count = pages.size() - page_start_index;
  if (page_count > 0) {
    auto& spans = entry.fixed_width_chunk_page_spans[column_name];
    if (spans.size() <= chunk_index) { spans.resize(chunk_index + 1); }
    // 0 tells find_covering_fixed_page the stride is not constant.
    spans[chunk_index] = fixed_width_chunk_page_span{
      page_start_index, page_count, boundaries.empty() ? rows_per_page : std::size_t{0}};
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

/// Add columns this entry does not yet have, at an EXISTING chunk index.
///
/// Used when projections share one entry: a later query over the same row groups
/// brings different columns, and they belong to the chunk that already holds those
/// rows rather than to a new one. Columns already present are skipped -- re-adding
/// one would append a second page list for the same rows and double the column's
/// contribution to every later batch.
///
/// Returns the number of pages added.
std::size_t merge_columns_into_chunk(
  pinned_entry& entry,
  std::string const& name,
  std::vector<std::string> const& column_names,
  std::vector<bool> const& fixed_columns,
  cudf::table_view const& view,
  std::size_t chunk_index,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream,
  std::unordered_map<std::string, std::shared_ptr<cudf::column>>* dictionaries)
{
  auto const pages_before = fixed_width_page_count(entry);
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    auto const column = std::string{column_names[i]};
    if (entry.fixed_width_pages_by_column.contains(column) ||
        entry.data_batches_by_column.contains(column)) {
      continue;  // already held for every chunk; nothing to add
    }
    auto const column_view = view.column(static_cast<cudf::size_type>(i));

    if (!fixed_columns[i]) {
      if (dictionary_encoding_enabled() && column_view.type().id() == cudf::type_id::STRING &&
          column_view.size() > 0) {
        if (auto codes = encode_column_as_dictionary(
              entry, name, column, column_view, memory_space, stream, dictionaries)) {
          index_fixed_width_column_pages(entry,
                                         name,
                                         entry.cache_info.resolved_file_paths.empty()
                                           ? std::string{}
                                           : entry.cache_info.resolved_file_paths.front(),
                                         column,
                                         codes->view(),
                                         chunk_index,
                                         0,
                                         &memory_space,
                                         entry.fixed_width_page_size_bytes,
                                         true);
          entry.dictionary_encoded_columns.insert(column);
          entry.data_batches_by_column[column].emplace_back(nullptr);
          entry.cache_info.names.push_back(column);
          continue;
        }
      }
      if (is_intrinsically_fixed_width_type(column_view)) { continue; }
      entry.data_batches_by_column[column].emplace_back(
        std::make_shared<cudf::column>(column_view, stream, memory_space.get_default_allocator()));
      entry.cache_info.names.push_back(column);
      continue;
    }

    index_fixed_width_column_pages(entry,
                                   name,
                                   entry.cache_info.resolved_file_paths.empty()
                                     ? std::string{}
                                     : entry.cache_info.resolved_file_paths.front(),
                                   column,
                                   column_view,
                                   chunk_index,
                                   0,
                                   &memory_space,
                                   entry.fixed_width_page_size_bytes,
                                   true);
    entry.data_batches_by_column[column].emplace_back(nullptr);
    entry.cache_info.names.push_back(column);
  }
  auto const after = fixed_width_page_count(entry);
  return after > pages_before ? after - pages_before : 0;
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
    // The actual device_buffer free below (via owned_column.reset()) must run with
    // this page's owning device current -- cuMemFreeAsync resolves the pool/context
    // from the calling thread's current device, not from the stream captured in the
    // buffer, and this function can be invoked from a worker thread whose current
    // device is left over from unrelated prior work.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
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

/// Per-device backoff state for apply_global_page_cache_memory_pressure's
/// circuit breaker below. Every caller of that function holds
/// _pinned_entries_mutex for its whole duration (verified: all 4 call sites
/// take the lock before reaching here), so a plain non-atomic map is safe.
struct reactive_pressure_backoff_state {
  std::size_t consecutive_noop_evictions{0};
  std::size_t skip_remaining{0};
};
std::unordered_map<int, reactive_pressure_backoff_state> g_reactive_pressure_backoff;

/// Reactive memory-pressure eviction across BOTH fixed-width and
/// variable-width page caches, sharing ONE cudaMemGetInfo probe per device
/// instead of two. Before this merge, fixed-width and variable-width each had
/// their own reactive-pressure function, called back-to-back at every one of
/// their 4 shared call sites -- each independently issuing a blocking
/// cudaMemGetInfo call against the same real GPU free memory, doubling that
/// driver-call cost on every insert for any table with cached STRING columns.
/// Candidates from both page types are pooled and evicted in true cross-type
/// LRU order (coldest last_access_tick first, regardless of type) rather than
/// fully draining one type before touching the other.
///
/// Circuit breaker: RMM's pool/async allocators generally do NOT return freed
/// device memory to the driver's free-memory view immediately (or sometimes
/// ever, while the pool itself is still live) -- so cudaMemGetInfo's reading
/// can stay pinned below min_free even after a real eviction, particularly
/// once the page cache's own configured budget is large enough that the pool
/// has grown to claim most of the device. Without a backoff, this function
/// then re-fires on literally every subsequent insert, re-scanning and
/// re-sorting every resident page and evicting more of them for zero
/// measurable benefit -- reproduced directly: raising the page cache budget
/// on this benchmark took free_bytes_before/free_bytes_after from "eviction
/// helps sometimes" to byte-for-byte identical on every call, 400+ times in a
/// single 10x run, and made the whole condition slower than baseline.
/// Detecting "evicted something but free memory didn't measurably improve"
/// and skipping the NEXT several calls (exponential, capped) for that device
/// converts that wasted, unbounded work into a bounded, self-limiting cost.
void apply_global_page_cache_memory_pressure(
  std::unordered_map<std::string, pinned_entry>& entries,
  std::string_view reason)
{
  auto const min_free = fixed_page_cache_min_free_bytes_per_gpu();
  if (min_free == 0) { return; }

  // Cheap pass first: probe every visible device's real free memory (and
  // check backoff) BEFORE paying for the expensive per-entry/per-page
  // candidate collection below. This function runs on every relevant cache
  // insert (several times per query, and its cost was proportional to total
  // resident pages across every entry) -- walking the whole cache just to
  // discover "nothing needs evicting", the common case since the proactive
  // budget passes already keep things comfortably under min_free most of the
  // time, was pure waste that grew with cache size over a long run.
  // cudaMemGetInfo itself is genuinely cheap (~20us, confirmed via nsys) --
  // the collection below, not this probe, was the actual cost.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) { return; }

  std::unordered_map<int, std::size_t> free_before_by_device;
  std::size_t total_bytes = 0;
  for (int device_id = 0; device_id < device_count; ++device_id) {
    auto& backoff = g_reactive_pressure_backoff[device_id];
    if (backoff.skip_remaining > 0) {
      --backoff.skip_remaining;
      continue;
    }
    std::size_t free_bytes = 0;
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
    auto const status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      SIRIUS_LOG_WARN(
        "[page-cache] memory_pressure cudaMemGetInfo failed device={} error='{}'",
        device_id,
        cudaGetErrorString(status));
      continue;
    }
    if (free_bytes < min_free) { free_before_by_device[device_id] = free_bytes; }
  }
  if (free_before_by_device.empty()) { return; }

  struct eviction_candidate {
    pinned_entry* entry{nullptr};  // owning entry: metrics for fixed pages, drain unit for variable
    fixed_width_column_page* fixed_page{nullptr};
    variable_width_column_page* variable_page{nullptr};
    std::uint64_t last_access_tick{0};
  };
  std::unordered_map<int, std::vector<eviction_candidate>> candidates_by_device;

  for (auto& [entry_name, entry] : entries) {
    for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
      for (auto& page : col_pages) {
        if (page.state != fixed_width_page_state::resident || page.key.device_id < 0 ||
            !page.owned_column || fixed_width_page_is_active(page) ||
            !free_before_by_device.contains(page.key.device_id)) {
          continue;
        }
        candidates_by_device[page.key.device_id].push_back(
          eviction_candidate{&entry,
                             &page,
                             nullptr,
                             page.last_access_tick.load(std::memory_order_relaxed)});
      }
    }
    for (auto& [_, chunk_indices] : entry.variable_width_pages_by_column) {
      for (auto& chunk_idx : chunk_indices) {
        for (auto& page : chunk_idx.pages) {
          if (page.state != variable_width_page_state::resident || !page.owned_column ||
              page.owned_column.use_count() > 1 || page.memory_space == nullptr) {
            continue;
          }
          auto const device_id = page.memory_space->get_device_id();
          if (device_id < 0 || !free_before_by_device.contains(device_id)) { continue; }
          candidates_by_device[device_id].push_back(
            eviction_candidate{&entry,
                               nullptr,
                               &page,
                               page.last_access_tick.load(std::memory_order_relaxed)});
        }
      }
    }
  }

  for (auto& [device_id, free_before] : free_before_by_device) {
    auto& backoff = g_reactive_pressure_backoff[device_id];
    auto candidates_it = candidates_by_device.find(device_id);
    if (candidates_it == candidates_by_device.end()) { continue; }
    auto& candidates = candidates_it->second;

    std::sort(candidates.begin(), candidates.end(), [](auto const& lhs, auto const& rhs) {
      return lhs.last_access_tick < rhs.last_access_tick;
    });

    auto const required_bytes = min_free - free_before;
    std::size_t released_alloc_bytes  = 0;
    std::size_t evicted_fixed_pages   = 0;
    std::size_t evicted_fixed_bytes   = 0;
    std::size_t evicted_variable_pages = 0;
    std::size_t evicted_variable_bytes = 0;
    std::uint64_t min_evicted_tick = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_evicted_tick = 0;
    /// Entries whose variable pages have already been drained this pass, so a
    /// second candidate page from the same entry is a no-op instead of a
    /// re-scan.
    std::unordered_set<pinned_entry*> drained_entries;
    // See the matching comment in apply_global_fixed_width_page_budget: the actual
    // free below needs this page's owning device current on this thread.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
    for (auto const& candidate : candidates) {
      if (released_alloc_bytes >= required_bytes) { break; }
      if (candidate.fixed_page) {
        auto& page = *candidate.fixed_page;
        if (page.state != fixed_width_page_state::resident || !page.owned_column ||
            fixed_width_page_is_active(page)) {
          continue;
        }
        auto const alloc_bytes = fixed_width_page_resident_alloc_bytes(page);
        page.state = fixed_width_page_state::evicted;
        page.owned_column.reset();
        released_alloc_bytes += alloc_bytes;
        evicted_fixed_bytes += page.num_bytes;
        ++candidate.entry->fixed_width_page_metrics.eviction_count;
        ++evicted_fixed_pages;
      } else if (candidate.variable_page) {
        // Variable pages evict at entry granularity, for the same reason
        // apply_global_variable_width_page_budget does: a paged column has no
        // whole-chunk fallback, so dropping one page makes the entire entry
        // unservable. Freeing only part of it would strand the rest as memory
        // that can no longer produce a cache hit.
        if (!candidate.entry || !drained_entries.insert(candidate.entry).second) { continue; }
        auto const pages_before = evicted_variable_pages;
        for (auto& [_, chunk_indices] : candidate.entry->variable_width_pages_by_column) {
          for (auto& chunk_idx : chunk_indices) {
            for (auto& page : chunk_idx.pages) {
              if (page.state != variable_width_page_state::resident || !page.owned_column ||
                  page.owned_column.use_count() > 1) {
                continue;
              }
              page.state = variable_width_page_state::evicted;
              page.owned_column.reset();
              released_alloc_bytes += page.num_bytes;
              evicted_variable_bytes += page.num_bytes;
              ++evicted_variable_pages;
            }
          }
        }
        // Nothing in this entry was actually evictable -- don't record a tick for it.
        if (evicted_variable_pages == pages_before) { continue; }
      } else {
        continue;
      }
      min_evicted_tick = std::min(min_evicted_tick, candidate.last_access_tick);
      max_evicted_tick = std::max(max_evicted_tick, candidate.last_access_tick);
    }

    if (evicted_fixed_pages == 0 && evicted_variable_pages == 0) {
      // Pressure detected but nothing was evictable (all candidates active/in-use) --
      // retrying on the very next insert won't change that. Same backoff as a
      // no-progress eviction below.
      backoff.consecutive_noop_evictions =
        std::min<std::size_t>(backoff.consecutive_noop_evictions + 1, 6);
      backoff.skip_remaining = std::size_t{1} << backoff.consecutive_noop_evictions;
      continue;
    }

    std::size_t free_after = 0;
    {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
      auto const sync_status = cudaDeviceSynchronize();
      if (sync_status != cudaSuccess) {
        SIRIUS_LOG_WARN(
          "[page-cache] memory_pressure cudaDeviceSynchronize failed device={} error='{}'",
          device_id,
          cudaGetErrorString(sync_status));
      }
      auto const info_status = cudaMemGetInfo(&free_after, &total_bytes);
      if (info_status != cudaSuccess) {
        SIRIUS_LOG_WARN(
          "[page-cache] memory_pressure post-evict cudaMemGetInfo failed device={} error='{}'",
          device_id,
          cudaGetErrorString(info_status));
        free_after = 0;
      }
    }

    if (free_after > free_before) {
      backoff.consecutive_noop_evictions = 0;
      backoff.skip_remaining             = 0;
    } else {
      // Evicted real pages but cudaMemGetInfo shows no improvement -- almost
      // certainly the pool allocator retaining freed memory rather than
      // returning it to the driver. Back off exponentially (capped at 64
      // calls) instead of repeating this same wasted work on every insert.
      backoff.consecutive_noop_evictions =
        std::min<std::size_t>(backoff.consecutive_noop_evictions + 1, 6);
      backoff.skip_remaining = std::size_t{1} << backoff.consecutive_noop_evictions;
    }

    SIRIUS_LOG_INFO(
      "[page-cache] memory_pressure applied scope=global reason='{}' device={} policy=lru "
      "min_free_bytes_per_gpu={} free_bytes_before={} free_bytes_after={} total_bytes={} "
      "evicted_fixed_pages={} evicted_fixed_bytes={} evicted_variable_pages={} "
      "evicted_variable_bytes={} evicted_alloc_bytes={} min_last_access_tick={} "
      "max_last_access_tick={}",
      reason,
      device_id,
      min_free,
      free_before,
      free_after,
      total_bytes,
      evicted_fixed_pages,
      evicted_fixed_bytes,
      evicted_variable_pages,
      evicted_variable_bytes,
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

void apply_global_variable_width_page_budget(std::unordered_map<std::string, pinned_entry>& entries);

/// Reports how much device memory the allocator actually holds against how much the
/// page cache believes it holds. The gap is what the pool has reserved but cannot
/// hand back -- i.e. external fragmentation plus pool slack, which is the quantity
/// uniform slab allocation exists to drive to zero. Throttled because it issues a
/// cudaMemGetInfo per device.
void report_page_cache_memory_overhead(
  std::unordered_map<std::string, pinned_entry> const& entries)
{
  static std::atomic<std::uint64_t> calls{0};
  auto const n = calls.fetch_add(1, std::memory_order_relaxed);
  if (n % 64 != 0) { return; }

  std::size_t live_fixed = 0;
  std::size_t live_variable = 0;
  std::size_t fixed_pages = 0;
  std::size_t variable_pages = 0;
  for (auto const& [_, entry] : entries) {
    for (auto const& [_, pages] : entry.fixed_width_pages_by_column) {
      for (auto const& page : pages) {
        if (page.state == fixed_width_page_state::resident && page.owned_column) {
          live_fixed += fixed_width_page_resident_alloc_bytes(page);
          ++fixed_pages;
        }
      }
    }
    for (auto const& [_, chunk_indices] : entry.variable_width_pages_by_column) {
      for (auto const& chunk_idx : chunk_indices) {
        for (auto const& page : chunk_idx.pages) {
          if (page.state == variable_width_page_state::resident && page.owned_column) {
            live_variable += page.num_bytes;
            ++variable_pages;
          }
        }
      }
    }
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) { return; }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { return; }
  }
  auto const device_used = total_bytes - free_bytes;
  auto const live        = live_fixed + live_variable;

  // Real pool accounting. cudaMemGetInfo alone cannot separate fragmentation from
  // retention: rmm's cuda_async_memory_resource leaves the release threshold at
  // UINT64_MAX, so the driver's view is a high-water mark that never falls.
  // pool_used is what the pool currently hands out; pool_reserved is what it holds
  // from the driver. reserved - used IS the external fragmentation plus slack, and
  // it is the number uniform slab allocation exists to shrink.
  std::size_t pool_used = 0;
  std::size_t pool_reserved = 0;
  for (auto const& [_, entry] : entries) {
    if (entry.memory_space == nullptr) { continue; }
    auto pool = entry.memory_space->get_pool_handle();
    if (pool == nullptr) { continue; }
    unsigned long long u = 0;
    unsigned long long r = 0;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &u);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &r);
    pool_used     = static_cast<std::size_t>(u);
    pool_reserved = static_cast<std::size_t>(r);
    break;  // single-GPU prototype: one pool
  }

  SIRIUS_LOG_INFO(
    "[page-cache] mem_overhead device_used_bytes={} cache_live_bytes={} gap_bytes={} "
    "pool_used_bytes={} pool_reserved_bytes={} pool_frag_bytes={} "
    "fixed_live={} fixed_pages={} variable_live={} variable_pages={}",
    device_used,
    live,
    device_used > live ? device_used - live : 0,
    pool_used,
    pool_reserved,
    pool_reserved > pool_used ? pool_reserved - pool_used : 0,
    live_fixed,
    fixed_pages,
    live_variable,
    variable_pages);
}

void apply_global_fixed_width_page_eviction_policies(
  std::unordered_map<std::string, pinned_entry>& entries,
  std::string_view reason)
{
  apply_global_fixed_width_page_budget(entries);
  apply_global_variable_width_page_budget(entries);
  apply_global_page_cache_memory_pressure(entries, reason);
  report_page_cache_memory_overhead(entries);
}

std::size_t variable_width_page_cache_budget_bytes_per_gpu()
{
  return parse_byte_size_or_zero(std::getenv("SIRIUS_VARIABLE_PAGE_CACHE_BYTES_PER_GPU"));
}

/// Global LRU eviction policy for variable-width pages, operating at
/// **pinned-entry granularity** rather than per page.
///
/// Per-page LRU is the obvious design and it is actively harmful here, because
/// the read path cannot serve a partially resident entry:
/// fixed_page_databatch_provider::usable() demands `_covered_rows ==
/// entry.num_rows`, and insert_fixed_page_entry_from_view deliberately stores
/// nullptr in data_batches_by_column for a successfully paged column (skipping
/// the redundant whole-chunk copy), so a paged column has no chunk-level
/// fallback to drop back to. Evicting a single variable page therefore does not
/// shrink an entry -- it *poisons* it: every column of that entry, including
/// fully resident fixed-width pages, stops being servable and the next hit
/// falls all the way through to a rescan from disk.
///
/// That is not hypothetical. On the SF50 10x benchmark the per-page version
/// evicted 286 pages across 78 budget events, and the run went from 354 cache
/// hits (fixed-width paging alone, zero provider failures) to 195 hits with 158
/// "page-only storage with incomplete resident coverage" fall-throughs -- a 45%
/// loss of cache hits, and the whole of that condition's ~5.9s regression
/// against fixed-width paging alone. The cost was never in indexing (measured
/// 201ms cumulative), reading (87ms) or the eviction scan itself (9ms).
///
/// Evicting the coldest *entry* wholesale keeps every surviving entry at 100%
/// coverage, so the cache holds fewer things but every one of them can still be
/// hit. Entries with any page currently in use (owned_column.use_count() > 1)
/// are skipped, same as before.
void apply_global_variable_width_page_budget(std::unordered_map<std::string, pinned_entry>& entries)
{
  auto const budget = variable_width_page_cache_budget_bytes_per_gpu();
  if (budget == 0) { return; }
  constexpr std::size_t headroom = 512ULL * 1024ULL * 1024ULL;
  auto const target_budget       = budget > headroom ? budget - headroom : budget;

  // Cheap pass first: just sum resident bytes, no candidate vector. This
  // function runs on every relevant cache insert; the common case (resident
  // bytes already comfortably under budget) needs nothing beyond this sum.
  // Count each device buffer ONCE. Entries that adopted a shared column-chunk all
  // point at the same allocation, so summing per entry triple-counts it and the
  // sweep evicts against a resident figure several times the real one -- the same
  // class of bug as an eviction pass that cannot move its own metric.
  std::size_t resident_bytes = 0;
  std::unordered_set<cudf::column const*> counted;
  for (auto& [_, entry] : entries) {
    for (auto& [_, chunk_indices] : entry.variable_width_pages_by_column) {
      for (auto& chunk_idx : chunk_indices) {
        for (auto& page : chunk_idx.pages) {
          if (page.state == variable_width_page_state::resident && page.owned_column &&
              counted.insert(page.owned_column.get()).second) {
            resident_bytes += page.num_bytes;
          }
        }
      }
    }
  }
  if (resident_bytes <= target_budget) { return; }

  // Over budget -- collect whole entries as the eviction unit. An entry's
  // recency is its *newest* page tick: one page read recently means the entry
  // as a whole was hit recently, and dropping it would throw away a live hit.
  struct entry_candidate {
    pinned_entry* entry{nullptr};
    std::size_t resident_bytes{0};
    std::uint64_t newest_tick{0};
  };
  std::vector<entry_candidate> candidates;
  for (auto& [_, entry] : entries) {
    entry_candidate candidate{&entry, 0, 0};
    bool in_use = false;
    for (auto& [_, chunk_indices] : entry.variable_width_pages_by_column) {
      for (auto& chunk_idx : chunk_indices) {
        for (auto& page : chunk_idx.pages) {
          if (page.state != variable_width_page_state::resident || !page.owned_column) {
            continue;
          }
          // A second reference means a materialized batch is holding this page
          // right now; resetting here would not free the memory anyway, and
          // would make the page permanently unreachable as a future hit.
          if (page.owned_column.use_count() > 1) {
            in_use = true;
            break;
          }
          candidate.resident_bytes += page.num_bytes;
          candidate.newest_tick =
            std::max(candidate.newest_tick, page.last_access_tick.load(std::memory_order_relaxed));
        }
        if (in_use) { break; }
      }
      if (in_use) { break; }
    }
    if (!in_use && candidate.resident_bytes > 0) { candidates.push_back(candidate); }
  }

  std::sort(candidates.begin(), candidates.end(), [](auto const& a, auto const& b) {
    return a.newest_tick < b.newest_tick;
  });

  std::size_t evicted_entries = 0;
  std::size_t evicted_pages   = 0;
  std::size_t evicted_bytes   = 0;
  for (auto& candidate : candidates) {
    if (resident_bytes <= target_budget) { break; }
    for (auto& [_, chunk_indices] : candidate.entry->variable_width_pages_by_column) {
      for (auto& chunk_idx : chunk_indices) {
        for (auto& page : chunk_idx.pages) {
          if (page.state != variable_width_page_state::resident || !page.owned_column) {
            continue;
          }
          page.state = variable_width_page_state::evicted;
          page.owned_column.reset();
          resident_bytes -= std::min(resident_bytes, page.num_bytes);
          evicted_bytes += page.num_bytes;
          ++evicted_pages;
        }
      }
    }
    ++evicted_entries;
  }

  if (evicted_pages != 0) {
    SIRIUS_LOG_INFO(
      "[variable-page-cache] page_budget applied scope=global policy=lru-entry "
      "budget_bytes_per_gpu={} target_bytes_per_gpu={} evicted_entries={} evicted_pages={} "
      "evicted_bytes={} resident_bytes_after={}",
      budget,
      target_budget,
      evicted_entries,
      evicted_pages,
      evicted_bytes,
      resident_bytes);
  }
}

// wdy end
}  // namespace

std::size_t sirius_scan_manager::fixed_page_admission_limit_bytes()
{
  return fixed_page_admission_max_entry_bytes();
}

std::size_t sirius_scan_manager::fixed_page_cache_budget_bytes()
{
  return fixed_page_cache_budget_bytes_per_gpu();
}

bool sirius_scan_manager::variable_width_page_cache_is_enabled()
{
  return variable_width_page_cache_enabled();
}

bool sirius_scan_manager::column_keyed_cache_is_enabled() { return column_keyed_cache_enabled(); }

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
  g_page_store_owner.store(this, std::memory_order_release);
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
  g_page_store_owner.store(nullptr, std::memory_order_release);
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
  apply_global_page_cache_memory_pressure(_pinned_entries, reason);
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
    // On a FULL pinned-cache hit the coalescer serves this operator from the
    // cached batch_provider (process_cached_entries) and no read is issued, so the
    // disk-reading split_provider is skipped entirely.
    //
    // On a PARTIAL hit it must still be built: the ingestible has been told which
    // row groups the cache covers and will read only the complement, and
    // process_provider_inputs blocks on a queue that only the split provider
    // feeds. Skipping it there hangs the pipeline -- the queue never fills and
    // nobody closes the connector.
    bool residual_scan = false;
    if (try_assign_cached_entries(op, &residual_scan) && !residual_scan) {
      // Full hit: nothing to read, and no split_provider to register.
      _scan_op_order.push_back(op);
      continue;
    }
    // Miss, or partial hit: fall through and build the split_provider. The push
    // into _scan_op_order happens once, below -- pushing here as well made
    // start_metadata_processing run the same provider twice.
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
  _admission_suspended.store(false, std::memory_order_release);
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

namespace {

/// Is [clo,chi] contained in [plo,phi], honouring open/closed ends?
bool range_within(cache_filter_range const& consumer, cache_filter_range const& producer)
{
  if (producer.has_lo) {
    if (!consumer.has_lo) { return false; }
    if (consumer.lo < producer.lo) { return false; }
    // A closed consumer end sitting exactly on an open producer end is outside it.
    if (consumer.lo == producer.lo && consumer.lo_inclusive && !producer.lo_inclusive) {
      return false;
    }
  }
  if (producer.has_hi) {
    if (!consumer.has_hi) { return false; }
    if (producer.hi < consumer.hi) { return false; }
    if (consumer.hi == producer.hi && consumer.hi_inclusive && !producer.hi_inclusive) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool filter_ranges_subsume(std::vector<cache_filter_range> const& producer,
                           bool producer_analyzable,
                           std::vector<cache_filter_range> const& consumer,
                           bool consumer_analyzable)
{
  if (!producer_analyzable || !consumer_analyzable) { return false; }
  if (producer.empty()) { return true; }  // producer kept everything
  for (auto const& p : producer) {
    bool covered = false;
    for (auto const& c : consumer) {
      if (c.column_name != p.column_name) { continue; }
      if (range_within(c, p)) {
        covered = true;
        break;
      }
    }
    if (!covered) { return false; }
  }
  return true;
}

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

std::size_t sirius_scan_manager::insert_fixed_page_entry_from_view(
  const std::string& name,
  cache_entry_info cache_info,
  cudf::table_view view,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream,
  bool fixed_width_pre_rejected,
  chunk_provenance provenance)
{
  (void)stream;
  if (view.num_columns() <= 0 || view.num_rows() <= 0) { return 0; }

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
  bool const hybrid_provider          = fixed_page_hybrid_provider_enabled();
  bool const variable_width_enabled   = variable_width_page_cache_enabled();
  std::vector<bool> fixed_columns;
  fixed_columns.reserve(static_cast<std::size_t>(view.num_columns()));
  bool has_fixed_width_column       = false;
  bool has_variable_width_candidate = false;
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    // A pre-rejected entry stores no fixed-width pages at all: marking the column
    // non-fixed routes it to the `is_intrinsically_fixed_width_type` branch below,
    // which records a nullptr chunk and copies nothing, while leaving STRING columns
    // to the variable-width index untouched.
    bool const fixed_width =
      !fixed_width_pre_rejected && is_fixed_width_page_candidate(view.column(i));
    fixed_columns.push_back(fixed_width);
    has_fixed_width_column = has_fixed_width_column || fixed_width;
    if (!fixed_width && variable_width_enabled &&
        view.column(i).type().id() == cudf::type_id::STRING) {
      has_variable_width_candidate = true;
    }
    if (!fixed_width && !hybrid_provider) {
      throw std::invalid_argument(
        "[sirius_scan_manager::insert_fixed_page_entry_from_view] non fixed-width column");
    }
  }
  // Previously just !has_fixed_width_column -- widened so a batch made up
  // entirely of STRING columns (e.g. SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0,
  // isolating the variable-width path for an experiment) still reaches the
  // per-column loop below instead of bailing out before
  // index_variable_width_columns_for_chunk ever runs.
  if (!has_fixed_width_column && !has_variable_width_candidate) { return 0; }

  std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
  apply_global_page_cache_memory_pressure(_pinned_entries, "pre_fixed_page_insert");

  // Pre-rejected entries contribute no fixed-width bytes, so the size-aware
  // admission check has nothing to weigh -- running it would reject and blacklist
  // the name, destroying the variable-width pages this insert exists to build.
  auto const admission_limit =
    fixed_width_pre_rejected ? std::size_t{0} : fixed_page_admission_max_entry_bytes();
  auto const incoming_bytes =
    fixed_width_pre_rejected ? std::size_t{0} : fixed_width_table_view_bytes(view);
  auto reject_admission = [&](std::size_t existing_bytes, std::size_t projected_bytes) -> std::size_t {
    // Column-keyed mode has ONE entry per (file, filter), so erasing it on an
    // oversized insert throws away every column already cached and blacklists the
    // name for the rest of the run -- measured as hits 49 -> 21. There the right
    // response is to stop widening the entry, not to destroy it: the columns
    // already resident keep serving the queries that ask for them, because
    // build_ranges and has_chunk_backing_for_selected_columns only ever look at
    // the columns a query actually selected.
    if (column_keyed_cache_enabled()) {
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] admission_stop_widening table='{}' existing_bytes={} "
        "incoming_bytes={} projected_bytes={} max_entry_bytes={}",
        name, existing_bytes, incoming_bytes, projected_bytes, admission_limit);
      return 0;
    }
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
    return 0;
  };

  if (!column_keyed_cache_enabled() && _fixed_page_admission_rejected_entries.contains(name)) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] auto_cache_skip reason=admission_previously_rejected table='{}' "
      "incoming_bytes={} max_entry_bytes={}",
      name,
      incoming_bytes,
      admission_limit);
    return 0;
  }

  if (admission_limit != 0 && incoming_bytes > admission_limit) {
    return reject_admission(0, incoming_bytes);
  }

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    auto& entry = existing_it->second;
    // Column-keyed mode merges projections into one entry, so a differing column
    // set is expected rather than a reason to erase what is already cached.
    bool const appendable =
      is_auto_fixed_page_entry(name) &&
      entry.cache_info.filter_signature == cache_info.filter_signature &&
      (column_keyed_cache_enabled() ||
       (entry.cache_info.column_ids.size() == cache_info.column_ids.size() &&
        entry.cache_info.names == column_names));
    if (!appendable) {
      orphan_or_erase_pinned_entry(_pinned_entries, existing_it);
      existing_it = _pinned_entries.end();
    } else if (column_keyed_cache_enabled() &&
               find_chunk_by_provenance(entry, provenance) !=
                 std::numeric_limits<std::size_t>::max() &&
               !std::ranges::all_of(column_names, [&entry](std::string_view col) {
                 return entry.data_batches_by_column.contains(std::string{col}) ||
                        entry.fixed_width_pages_by_column.contains(std::string{col});
               })) {
      // Merge: these row groups are already here, but this projection brings
      // columns the entry does not have yet. Add them AT THE EXISTING CHUNK INDEX
      // so a chunk index keeps meaning one range of the table for every column --
      // appending at a fresh index would let all_columns_cover_tiled assemble
      // columns covering different rows into one batch. num_rows is not touched:
      // the rows are already counted, only the column set widens.
      auto const chunk_index = find_chunk_by_provenance(entry, provenance);
      auto const added       = merge_columns_into_chunk(
        entry, name, column_names, fixed_columns, view, chunk_index, memory_space, stream,
        &_shared_dictionaries);
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] merged_columns table='{}' chunk_index={} added={} entry_columns={}",
        name, chunk_index, added, entry.cache_info.names.size());
      return added;
    } else if (chunk_already_present(entry, provenance)) {
      // Two concurrent scan tasks can both miss the cache for the same brand-new
      // (name, filter) key, both read it from disk, and both land here to populate
      // it. Without this guard the second call's "appendable" branch treats its own
      // from-scratch copy of the SAME rows as a new chunk and appends it, silently
      // doubling entry.num_rows and every future query's output for that table.
      //
      // The test used to be `view.num_rows() == entry.num_rows`, on the reasoning
      // that a real incremental chunk essentially never has a row count equal to
      // the running total. That holds only while chunks differ in size. ClickBench's
      // row groups are uniformly 10,000,000 rows, so the SECOND chunk of every
      // entry has exactly the running total's row count and was discarded as a
      // duplicate -- leaving entries permanently short (one stalled at 39,997,497
      // of 99,997,497 and was refused 103 times in a single run). Comparing the
      // row groups instead is exact and does not depend on chunk sizes differing.
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] auto_cache_skip reason=duplicate_concurrent_populate table='{}' "
        "incoming_rows={} existing_rows={}",
        name,
        view.num_rows(),
        entry.num_rows);
      return 0;
    } else {
      auto const existing_bytes  = fixed_width_entry_logical_bytes(entry);
      auto const projected_bytes = existing_bytes + incoming_bytes;
      if (admission_limit != 0 && projected_bytes > admission_limit) {
        return reject_admission(existing_bytes, projected_bytes);
      }
      // Whole-chunk copies share this bound. Splitting them out and bounding the
      // pair by (fixed + variable) budgets instead was measured to change nothing
      // at the default limit and to OOM once the pair could actually reach it, so
      // the single bound stays.
      auto const chunk_bytes = entry_whole_chunk_bytes(entry);
      if (admission_limit != 0 && projected_bytes + chunk_bytes > admission_limit) {
        return reject_admission(existing_bytes + chunk_bytes, projected_bytes + chunk_bytes);
      }
      auto const pages_before = fixed_width_page_count(entry);
      if (entry.fixed_width_page_size_bytes == 0) {
        entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();
      }
      auto const chunk_index = entry.chunk_memory_spaces.size();
      entry.chunk_memory_spaces.push_back(&memory_space);
      entry.chunk_provenance_by_index.push_back(provenance);
      auto const paged_columns = index_variable_width_columns_for_chunk(
        entry, name, column_names, fixed_columns, view, chunk_index, memory_space, stream,
        entry.cache_info.projected_column_bytes, &_shared_variable_pages);
      for (std::size_t i = 0; i < column_names.size(); ++i) {
        auto& chunks = entry.data_batches_by_column[std::string{column_names[i]}];
        auto const column_view = view.column(static_cast<cudf::size_type>(i));
        if (!fixed_columns[i]) {
          bool const paged = paged_columns.contains(i);
          if (paged) {
            chunks.emplace_back(nullptr);  // paged copy is authoritative; skip the redundant whole-chunk copy
          } else if (is_intrinsically_fixed_width_type(column_view)) {
            chunks.emplace_back(nullptr);  // SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0: leave uncached
          } else if (!string_columns_cacheable()) {
            // Fixed-width-only condition: leave the STRING column out of the cache
            // entirely. Without this the "fixed-width cache" condition still stored
            // every STRING column as a whole chunk, so a run labelled fixed-only was
            // really fixed-width-paged plus string-whole-chunk -- and the comparison
            // against the variable-width cache measured a change of STORAGE FORMAT
            // for data that was cached either way, not the value of caching strings.
            // A query that needs this column then misses on has_chunk_backing and
            // reads it from parquet, which is the honest baseline.
            chunks.emplace_back(nullptr);
          } else {
            chunks.emplace_back(std::make_shared<cudf::column>(
              column_view, stream, memory_space.get_default_allocator()));
          }
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
      auto const pages_just_added = fixed_width_page_count(entry) - pages_before;
      entry.num_rows += static_cast<std::size_t>(view.num_rows());
      entry.fixed_width_page_metrics = {};
      apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
      auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
      entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
      entry.fixed_width_page_metrics.eviction_count = eviction_count;
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_directory direct_appended table='{}' fixed_cols={} pages={} "
        "pages_added={} resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} "
        "eviction_count={} directory_entries={} page_bytes={} rows={} owned_pages=1",
        name,
        entry.fixed_width_pages_by_column.size(),
        fixed_width_page_count(entry),
        pages_just_added,
        entry.fixed_width_page_metrics.resident_pages,
        entry.fixed_width_page_metrics.resident_bytes,
        entry.fixed_width_page_metrics.stats_pages,
        entry.fixed_width_page_metrics.evicted_pages,
        entry.fixed_width_page_metrics.eviction_count,
        entry.fixed_width_page_directory.size(),
        entry.fixed_width_page_size_bytes,
        entry.num_rows);
      return pages_just_added;
    }
  }

  pinned_entry entry;
  entry.cache_info = std::move(cache_info);
  entry.chunk_memory_spaces.push_back(&memory_space);
  entry.chunk_provenance_by_index.push_back(provenance);
  entry.tier                        = cucascade::memory::Tier::GPU;
  entry.num_rows                    = static_cast<std::size_t>(view.num_rows());
  entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();

  auto const paged_columns = index_variable_width_columns_for_chunk(
    entry, name, column_names, fixed_columns, view, /*chunk_index=*/0, memory_space, stream,
    entry.cache_info.projected_column_bytes, &_shared_variable_pages);
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    auto& chunks = entry.data_batches_by_column[std::string{column_names[i]}];
    auto const column_view = view.column(static_cast<cudf::size_type>(i));
    if (!fixed_columns[i]) {
      bool const paged = paged_columns.contains(i);
      // Dictionary path: a STRING column too large to cache as strings becomes one
      // int32 code per row plus a shared key set. The codes are fixed-width, so
      // from here down they are indistinguishable from any other int32 column --
      // same pages, same eviction, same O(1) lookup, no offsets overhead.
      if (!paged && dictionary_encoding_enabled() &&
          column_view.type().id() == cudf::type_id::STRING && column_view.size() > 0) {
        if (auto codes = encode_column_as_dictionary(
              entry, name, column_names[i], column_view, memory_space, stream,
              &_shared_dictionaries)) {
          index_fixed_width_column_pages(entry,
                                         name,
                                         entry.cache_info.resolved_file_paths.empty()
                                           ? std::string{}
                                           : entry.cache_info.resolved_file_paths.front(),
                                         column_names[i],
                                         codes->view(),
                                         /*chunk_index=*/0,
                                         0,
                                         &memory_space,
                                         entry.fixed_width_page_size_bytes,
                                         true);
          entry.dictionary_encoded_columns.insert(std::string{column_names[i]});
          chunks.emplace_back(nullptr);
          continue;
        }
      }
      if (paged) {
        chunks.emplace_back(nullptr);  // paged copy is authoritative; skip the redundant whole-chunk copy
      } else if (is_intrinsically_fixed_width_type(column_view)) {
        chunks.emplace_back(nullptr);  // SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0: leave uncached
      } else {
        chunks.emplace_back(std::make_shared<cudf::column>(
          column_view, stream, memory_space.get_default_allocator()));
      }
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

  auto const pages_just_added = fixed_width_page_count(entry);
  entry.fixed_width_page_metrics = {};
  _pinned_entries[name] = std::move(entry);
  auto& inserted_entry = _pinned_entries.at(name);
  apply_global_fixed_width_page_eviction_policies(_pinned_entries, "post_insert");
  auto eviction_count = inserted_entry.fixed_width_page_metrics.eviction_count;
  inserted_entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(inserted_entry);
  inserted_entry.fixed_width_page_metrics.eviction_count = eviction_count;
  // Column NAMES, not just the count: whether one entry per (file, predicate)
  // stores the same column several times over -- the cost of making the predicate
  // part of the cache identity rather than a tag on it -- cannot be read off a
  // count, and that duplication is the whole argument for merging the entries.
  std::vector<std::string> cached_columns;
  cached_columns.reserve(inserted_entry.fixed_width_pages_by_column.size());
  for (auto const& [column_name, pages] : inserted_entry.fixed_width_pages_by_column) {
    cached_columns.push_back(column_name);
  }
  std::sort(cached_columns.begin(), cached_columns.end());
  std::string column_list;
  for (auto const& column_name : cached_columns) { column_list += column_name + ","; }
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] page_directory direct_indexed table='{}' fixed_cols={} pages={} "
    "pages_added={} resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} "
    "eviction_count={} directory_entries={} page_bytes={} rows={} owned_pages=1 columns='{}'",
    name,
    inserted_entry.fixed_width_pages_by_column.size(),
    fixed_width_page_count(inserted_entry),
    pages_just_added,
    inserted_entry.fixed_width_page_metrics.resident_pages,
    inserted_entry.fixed_width_page_metrics.resident_bytes,
    inserted_entry.fixed_width_page_metrics.stats_pages,
    inserted_entry.fixed_width_page_metrics.evicted_pages,
    inserted_entry.fixed_width_page_metrics.eviction_count,
    inserted_entry.fixed_width_page_directory.size(),
    inserted_entry.fixed_width_page_size_bytes,
    inserted_entry.num_rows,
    column_list);
  return pages_just_added;
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
  apply_global_page_cache_memory_pressure(_pinned_entries, "pre_pinned_insert");

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
    orphan_or_erase_pinned_entry(_pinned_entries, existing_it);
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

/// The pages of @p column for @p chunk_index, concatenated back into the whole
/// column. Null unless every page is resident and they tile the chunk exactly --
/// a gap would silently shorten the result.
std::shared_ptr<cudf::column> concatenate_chunk_pages(pinned_entry const& entry,
                                                      std::string const& column,
                                                      std::size_t chunk_index,
                                                      rmm::cuda_stream_view stream)
{
  auto pages_it = entry.fixed_width_pages_by_column.find(column);
  if (pages_it == entry.fixed_width_pages_by_column.end()) { return nullptr; }
  auto const expected = chunk_index < entry.chunk_provenance_by_index.size()
                          ? entry.chunk_provenance_by_index[chunk_index].num_rows
                          : 0;
  if (expected == 0) { return nullptr; }

  std::vector<fixed_width_column_page const*> pages;
  for (auto const& page : pages_it->second) {
    if (page.chunk_index != chunk_index) { continue; }
    if (page.state != fixed_width_page_state::resident || !page.owned_column) { return nullptr; }
    pages.push_back(&page);
  }
  if (pages.empty()) { return nullptr; }
  std::sort(pages.begin(), pages.end(), [](auto const* a, auto const* b) {
    return a->row_offset < b->row_offset;
  });
  std::size_t cursor = 0;
  std::vector<cudf::column_view> views;
  views.reserve(pages.size());
  for (auto const* page : pages) {
    if (page->row_offset != cursor) { return nullptr; }  // hole: refuse rather than shorten
    views.push_back(page->owned_column->view());
    cursor += page->num_rows;
  }
  if (cursor != expected) { return nullptr; }
  if (views.size() == 1) { return pages.front()->owned_column; }
  return std::shared_ptr<cudf::column>(
    cudf::concatenate(views, stream, cudf::get_current_device_resource_ref()).release());
}

void sirius_scan_manager::insert_pages_from_view(
  std::string const& file_path,
  std::vector<cudf::size_type> const& row_groups,
  std::vector<std::size_t> const& row_group_rows,
  std::vector<std::string> const& column_names,
  cudf::table_view const& view,
  std::string const& filter_signature,
  std::vector<std::string> filter_conjuncts,
  std::vector<cache_filter_range> filter_ranges,
  bool filter_analyzable,
  cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream)
{
  // One page per (column, row group), so the row counts must actually partition the
  // view. They do not when the reader filtered: the view then holds fewer rows than
  // the row groups do, and there is no way to say which row group a surviving row
  // came from. Such a view is stored as a single page only when it covers exactly
  // one row group, where the answer is not in doubt.
  if (row_groups.empty() || column_names.empty()) { return; }
  std::size_t total = 0;
  for (auto const rows : row_group_rows) { total += rows; }
  bool const partitioned =
    row_group_rows.size() == row_groups.size() &&
    total == static_cast<std::size_t>(view.num_rows());
  if (!partitioned && row_groups.size() != 1) { return; }

  auto const budget = fixed_page_cache_budget_bytes_per_gpu();

  // Admission control, before anything is copied. Every page below is a GPU deep copy of a
  // column the query is holding right now, so an insert taken at the wrong moment hands the
  // cache a second copy of the query's own working set. ClickBench q24 is SELECT * over 105
  // columns, and with the cache on it died in the OOM retry loop where it succeeds with the
  // cache off: one row group offered 389 pages and the store sat at 5.6 GB against a 5.6 GB
  // budget while the query still needed every one of those columns. Evicting afterwards is
  // too late -- the copy has already been allocated -- so ask the same question the eviction
  // below asks, and if the pool is already short, cache nothing from this batch.
  //
  // The entry check alone is not enough, because it says nothing about how much THIS batch
  // is about to take: q24 passed it with 2.1 GB of headroom and then copied 6.44 GB in one
  // call, all 105 columns of a row group. So the same number also caps the copy -- admission
  // stops mid-batch once this call has taken the headroom it was given.
  if (_admission_suspended.load(std::memory_order_acquire)) {
    // This query already ran out of device memory once. Caching for it again would only
    // rebuild what was just thrown away and run it out again.
    return;
  }

  std::size_t admission_headroom = std::numeric_limits<std::size_t>::max();
  std::size_t admitted_bytes     = 0;
  bool admission_full            = false;
  if (auto const min_free = fixed_page_cache_min_free_bytes_per_gpu(); min_free > 0) {
    auto const free_bytes =
      _reservation_manager.get_available_memory_for_tier(cucascade::memory::Tier::GPU);
    if (free_bytes < min_free) {
      SIRIUS_LOG_INFO("[page-store] admission declined: free={} min_free={} columns={} "
                      "resident={}",
                      free_bytes,
                      min_free,
                      column_names.size(),
                      _pages_bytes);
      // Decline the copying, but fall through rather than return: the eviction at the end of
      // this function is what hands memory BACK, and a pool below its floor is exactly when
      // that is needed. Returning here would leave the store holding everything it had while
      // the query it is starving asks for more.
      admission_full = true;
    } else {
      admission_headroom = free_bytes - min_free;
    }
  }

  // The page store's mutex is deliberately NOT held across this loop. Every page here is a
  // GPU deep copy, and a STRING column's size is read back through the stream; holding the
  // mutex across those made every concurrent probe wait behind an insert. Measured on TPC-H
  // SF50: the scan breakdown charged 0.50s to the availability probe, of which 0.50s was
  // waiting for this mutex -- the hash lookups themselves are ~930 per scan and cost nothing.
  // So the lock is taken three times, each for bookkeeping only: to intern, to test whether
  // a page is already resident, and to publish the copy.
  std::size_t row_offset = 0;
  int file_id = 0;
  {
    std::lock_guard lock{_pages_mutex};
    file_id = intern_id(file_path);
  }
  for (std::size_t g = 0; g < row_groups.size() && !admission_full; ++g) {
    auto const rows = partitioned ? row_group_rows[g]
                                  : static_cast<std::size_t>(view.num_rows());
    if (rows == 0) { continue; }
    for (std::size_t c = 0; c < column_names.size(); ++c) {
      if (c >= static_cast<std::size_t>(view.num_columns())) { break; }
      auto const column_view = view.column(static_cast<cudf::size_type>(c));
      // STRING columns page exactly like fixed-width ones here. cached_page holds a
      // cudf::column, which is type-erased, and the serve path concatenates -- neither
      // cares about width. Keeping strings out sent them to a second cache with its own
      // budget, its own whole-entry eviction and a 512MiB per-column admission cap, so on
      // ClickBench (Title 9.9GB, URL 9.5GB) every string column was refused at the door
      // and 1GB of VRAM sat unused while the fixed-width side thrashed. One store, one
      // budget, one LRU: whichever pages are actually being read win the space.
      int column_id = 0;
      {
        std::lock_guard lock{_pages_mutex};
        column_id = intern_id(column_names[c]);
      }
      bool const fixed_width = is_intrinsically_fixed_width_type(column_view);
      if (!fixed_width && column_view.type().id() != cudf::type_id::STRING) { continue; }

      auto const rg_begin = static_cast<cudf::size_type>(row_offset);
      auto const rg_end   = static_cast<cudf::size_type>(row_offset + rows);

      // Cut the row group into pages of about fixed_width_page_size_bytes(). Paging
      // is the point of this cache: a whole row group of a wide column is far too
      // coarse a unit to keep or evict -- ClickBench's URL is 0.9GiB per row group,
      // 15% of a 6GB budget, so one eviction throws away 0.9GiB and residency can
      // only ever be a multiple of that. Rows per page come from the column's own
      // bytes per row: exact for fixed width, and for STRING from the run's chars
      // size (one device read of the end offsets, the same O(1) trick the
      // variable-width index uses) plus the 4-byte offset each row carries.
      double bytes_per_row = 0.0;
      if (fixed_width) {
        bytes_per_row = static_cast<double>(cudf::size_of(column_view.type()));
      } else {
        auto rg_slices = cudf::slice(column_view, {rg_begin, rg_end});
        if (rg_slices.empty()) { continue; }
        cudf::strings_column_view const scv{rg_slices.front()};
        bytes_per_row =
          (static_cast<double>(scv.chars_size(stream)) + 4.0 * static_cast<double>(rows)) /
          static_cast<double>(std::max<std::size_t>(1, rows));
      }
      if (bytes_per_row <= 0.0) { bytes_per_row = 1.0; }
      auto const target      = static_cast<double>(fixed_width_page_size_bytes());
      auto const rows_per_page = std::max<std::size_t>(
        1, static_cast<std::size_t>(target / bytes_per_row));

      auto const pages_in_row_group = (rows + rows_per_page - 1) / rows_per_page;
      // Keep one layout per row group. Rows-per-page is derived from this batch's average
      // row width, and for a STRING column that differs between batches, so a second pass
      // over the same row group can want more pages than the first. Those extra page
      // indices do not collide with what is already resident, so both layouts would end up
      // in the store overlapping each other. The pages already there are as good as these;
      // leave them alone rather than adding a second copy of the same rows.
      {
        std::lock_guard lock{_pages_mutex};
        auto const first =
          _pages.find(cached_page_key{file_id, column_id, static_cast<int>(row_groups[g]), 0});
        if (first != _pages.end() && first->second.pages_in_row_group != pages_in_row_group) {
          continue;
        }
      }
      int page_index                = 0;
      for (std::size_t r = 0; r < rows; r += rows_per_page, ++page_index) {
        auto const page_rows = std::min<std::size_t>(rows_per_page, rows - r);
        cached_page_key key{
          file_id, column_id, static_cast<int>(row_groups[g]), page_index};
        {
          std::lock_guard lock{_pages_mutex};
          if (_pages.contains(key)) { continue; }  // first writer wins; no duplicate copies
        }

        auto const begin = static_cast<cudf::size_type>(rg_begin + r);
        auto const end   = static_cast<cudf::size_type>(rg_begin + r + page_rows);
        auto slices      = cudf::slice(column_view, {begin, end});
        if (slices.empty()) { continue; }
        cached_page page;
        page.data = std::make_shared<cudf::column>(
          slices.front(), stream, space.get_default_allocator());
        page.num_rows          = page_rows;
        page.pages_in_row_group = pages_in_row_group;
        page.num_bytes         = page.data->alloc_size();
        if (admitted_bytes + page.num_bytes > admission_headroom) {
          // Taking this page would push the pool past the floor the query needs. Stop here:
          // the pages already published stay, the rest of this batch is simply not cached,
          // and the eviction below still runs so the store is left within its budget.
          SIRIUS_LOG_INFO("[page-store] admission capped: took={} headroom={} at column='{}'",
                          admitted_bytes,
                          admission_headroom,
                          column_names[c]);
          admission_full = true;
          break;
        }
        admitted_bytes += page.num_bytes;
        // Min/max page stats are a fixed-width notion (they drive predicate-based page
        // skipping). A STRING page carries none and is simply never skipped on stats.
        page.stats             = fixed_width
                                   ? compute_fixed_width_page_stats_for_range(column_view, begin, end, &space)
                                   : fixed_width_page_stats{};
        page.filter_signature  = filter_signature;
        page.filter_conjuncts  = filter_conjuncts;
        page.filter_ranges     = filter_ranges;
        page.filter_analyzable = filter_analyzable;
        std::lock_guard lock{_pages_mutex};
        // Another thread may have published this same page while the copy above ran; it
        // holds the same rows, so drop ours rather than pay for a second resident copy.
        if (_pages.contains(key)) { continue; }
        page.last_access_tick  = ++_page_tick;
        _lru.push_back(key);
        page.lru_it            = std::prev(_lru.end());
        _pages_bytes += page.num_bytes;
        _pages.emplace(std::move(key), std::move(page));
      }
      if (admission_full) { break; }
    }
    if (admission_full) { break; }
    row_offset += rows;
  }

  // LRU down to whichever bound binds first: the configured budget, or what the
  // device can actually spare.
  //
  // A budget alone is not enough. The cache competes with the query's own working
  // memory, and how much that needs is not knowable in advance -- ClickBench ran out
  // of device memory at 4.47 GB resident against a 6 GB budget, so the budget never
  // fired. Checking free device memory on every insert gives the cache a chance to
  // shrink DURING a query; the pre-existing memory-pressure sweep only ran at query
  // start, which is too late once a join has already taken the memory.
  //
  // Eviction does hold the mutex: it walks `_lru` and erases from `_pages`, so it has to be
  // atomic against a concurrent probe. It is one hold per insert call rather than one per
  // page, and it copies nothing.
  std::lock_guard evict_lock{_pages_mutex};
  std::size_t resident = _pages_bytes;

  std::size_t effective_budget = budget;
  auto const min_free          = fixed_page_cache_min_free_bytes_per_gpu();
  // Circuit breaker. RMM's pool does not hand freed device memory back to the
  // driver's view, so cudaMemGetInfo can stay pinned below min_free however much
  // this evicts. Without a backoff the check re-fires on every insert, re-sorts
  // every page and throws more away for no measurable gain -- documented on
  // apply_global_page_cache_memory_pressure, where it made the run slower than
  // baseline. Skip the next few checks whenever an eviction did not move the needle.
  if (_page_pressure_skip > 0) { --_page_pressure_skip; }
  if (min_free > 0 && _page_pressure_skip == 0) {
    // Sirius's own reservation manager, not cudaMemGetInfo. The engine reserves its
    // whole usage_limit_bytes up front, so the driver's view of free memory is
    // pinned at whatever was left after that reservation and does not move however
    // much the query or the cache actually uses -- measured: the check never fired
    // once while the run went on to exhaust device memory anyway. The reservation
    // manager reports what is free INSIDE that pool, which is what the query and
    // the cache are actually competing for.
    auto const free_bytes =
      _reservation_manager.get_available_memory_for_tier(cucascade::memory::Tier::GPU);
    if (free_bytes < min_free) {
      _page_pressure_free_before = free_bytes;
      // Give back what the device is short by, on top of any budget overshoot.
      auto const shortfall = min_free - free_bytes;
      auto const target    = resident > shortfall ? resident - shortfall : std::size_t{0};
      effective_budget     = (budget == 0) ? target : std::min(budget, target);
      SIRIUS_LOG_INFO(
        "[page-store] device pressure free={} min_free={} resident={} -> target={}",
        free_bytes,
        min_free,
        resident,
        effective_budget);
    }
  }
  if (effective_budget == 0 && budget == 0) { return; }
  auto const budget_to_use = effective_budget;
  if (resident <= budget_to_use) { return; }
  // Oldest first, straight off the front of the LRU list. No vector, no sort:
  // the list already holds exactly the order this needs.
  std::size_t evicted = 0;
  while (resident > budget_to_use && !_lru.empty()) {
    auto const key = _lru.front();
    auto it        = _pages.find(key);
    if (it == _pages.end()) {  // shouldn't happen; keep the list and map in step
      _lru.pop_front();
      continue;
    }
    resident -= it->second.num_bytes;
    _pages_bytes -= it->second.num_bytes;
    _pages.erase(it);
    _lru.pop_front();
    ++evicted;
  }
  if (evicted > 0) {
    SIRIUS_LOG_INFO("[page-store] evicted pages={} resident_bytes={} budget={}",
                    evicted,
                    resident,
                    budget_to_use);
    if (_page_pressure_free_before > 0) {
      auto const free_after =
        _reservation_manager.get_available_memory_for_tier(cucascade::memory::Tier::GPU);
      if (free_after <= _page_pressure_free_before) {
        _page_pressure_skip =
          _page_pressure_skip == 0 ? 4 : std::min<std::size_t>(_page_pressure_skip * 2, 64);
        SIRIUS_LOG_INFO("[page-store] pressure eviction freed nothing visible; skipping {}",
                        _page_pressure_skip);
      }
      _page_pressure_free_before = 0;
    }
  }
}

std::size_t sirius_scan_manager::drop_all_pages()
{
  auto const before =
    _reservation_manager.get_available_memory_for_tier(cucascade::memory::Tier::GPU);
  std::size_t freed      = 0;
  std::size_t still_held = 0;
  {
    std::lock_guard lock{_pages_mutex};
    freed = _pages_bytes;
    for (auto const& [_, page] : _pages) {
      // use_count 1 is this map's own reference; anything above it is somebody still reading
      // the page, and erasing the entry would not free its memory.
      if (page.data && page.data.use_count() > 1) { still_held += page.num_bytes; }
    }
    _pages.clear();
    _lru.clear();
    _pages_bytes = 0;
    // Let the pressure sweep try again immediately: its backoff assumes evicting does not
    // help, and dropping everything is a different proposition.
    _page_pressure_skip        = 0;
    _page_pressure_free_before = 0;
  }

  // The page store is not the only GPU-resident cache. Each pinned entry carries its own
  // whole-chunk columns and fixed- and variable-width page lists, and that is what a long run
  // accumulates: dropping only the page store left ClickBench q24 dying even after the drop
  // had handed back 6.4 GB, while the same drop in a two-query repro was enough.
  {
    std::lock_guard entries_lock{_pinned_entries_mutex};
    for (auto& [_, entry] : _pinned_entries) {
      entry.data_batches_by_column.clear();
      entry.fixed_width_pages_by_column.clear();
      entry.fixed_width_page_directory.clear();
      entry.fixed_width_chunk_page_spans.clear();
      entry.variable_width_pages_by_column.clear();
    }
  }

  auto const after =
    _reservation_manager.get_available_memory_for_tier(cucascade::memory::Tier::GPU);
  SIRIUS_LOG_WARN("[page-store] drop accounted={} still_referenced={} pool_free {} -> {} (+{})",
                  freed,
                  still_held,
                  before,
                  after,
                  after > before ? after - before : 0);
  _admission_suspended.store(true, std::memory_order_release);
  return freed;
}

std::size_t drop_page_store_on_oom()
{
  auto* owner = g_page_store_owner.load(std::memory_order_acquire);
  if (owner == nullptr) { return 0; }
  return owner->drop_all_pages();
}

std::pair<std::size_t, std::size_t> sirius_scan_manager::page_store_size() const
{
  std::lock_guard lock{_pages_mutex};
  std::size_t const bytes = _pages_bytes;  // maintained on insert and eviction
  return {bytes, _pages.size()};
}

int sirius_scan_manager::intern_id(std::string const& str)
{
  auto const [it, _] = _intern.try_emplace(str, static_cast<int>(_intern.size()));
  return it->second;
}

int sirius_scan_manager::intern_lookup(std::string const& str) const
{
  auto const it = _intern.find(str);
  return it == _intern.end() ? -1 : it->second;
}

std::vector<bool> sirius_scan_manager::cached_row_group_columns_available(
  std::string const& file_path,
  std::vector<cudf::size_type> const& row_groups,
  std::vector<std::string> const& column_names,
  std::string const& required_filter_signature,
  std::vector<std::string> const& required_filter_conjuncts)
{
  std::vector<bool> out(column_names.size(), false);
  if (row_groups.empty() || column_names.empty()) { return out; }

  // Conjunct subsumption was tried here and is WRONG for splicing. "The query's
  // rows are a subset of this page's rows" is not enough: the splice puts cached
  // columns side by side with columns the reader is about to produce, and the
  // reader applies the FULL predicate. A page filtered by `A` beside a column
  // filtered by `A AND B` holds more rows than its neighbour -- measured as
  // "Column size mismatch: 548873 != 558097" on ClickBench q38. The two halves
  // must hold exactly the SAME rows, so an identical predicate is a requirement,
  // not conservatism. Reuse under a narrower predicate needs the cached rows
  // re-filtered before they can be spliced, which this path does not do.
  (void)required_filter_conjuncts;

  // Split the wait for the page store from the work done inside it. The probe itself is
  // ~30 columns x 31 row groups of hash lookups, which cannot account for the seconds the
  // scan breakdown attributes to it; insert_pages_from_view holds this same mutex across a
  // per-page GPU deep copy, so a probe that lands mid-insert waits behind it.
  auto const lock_t0 = std::chrono::steady_clock::now();
  std::lock_guard lock{_pages_mutex};
  _page_lock_wait_us.fetch_add(
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - lock_t0)
      .count(),
    std::memory_order_relaxed);
  auto const file_id = intern_lookup(file_path);
  if (file_id < 0) { return out; }  // nothing from this file is cached
  for (std::size_t ci = 0; ci < column_names.size(); ++ci) {
    auto const column_id = intern_lookup(column_names[ci]);
    if (column_id < 0) { continue; }
    bool complete = true;
    for (auto const row_group : row_groups) {
      std::size_t pages_found = 0, pages_expected = 0;
      for (int page_index = 0;; ++page_index) {
        auto it = _pages.find(
          cached_page_key{file_id, column_id, static_cast<int>(row_group), page_index});
        if (it == _pages.end() || !it->second.data) { break; }
        // Same rule as the serve path: a run whose pages disagree on their page count is
        // two layouts of the same row group sharing a key space, and must not be spliced.
        if (pages_expected == 0) {
          pages_expected = it->second.pages_in_row_group;
        } else if (pages_expected != it->second.pages_in_row_group) {
          complete = false;
          break;
        }
        if (!it->second.filter_signature.empty() &&
            it->second.filter_signature != required_filter_signature) {
          complete = false;
          break;
        }
        if (it->second.filter_signature.empty() && !required_filter_signature.empty()) {
          complete = false;
          break;
        }
        ++pages_found;
      }
      if (!complete || pages_found == 0 || pages_found != pages_expected) {
        complete = false;
        break;
      }
    }
    out[ci] = complete;
  }
  return out;
}

std::vector<std::shared_ptr<cudf::column>> sirius_scan_manager::cached_row_group_columns(
  std::string const& file_path,
  std::vector<cudf::size_type> const& row_groups,
  std::vector<std::string> const& column_names,
  rmm::cuda_stream_view stream,
  std::string const& required_filter_signature,
  std::vector<std::string> const& required_filter_conjuncts)
{
  // A page serves this request when its predicate keeps at least the rows the
  // request wants. Exact text match is one way; the other is that the page's
  // AND-ed parts are all present in the request's, which means the request only
  // narrows further. Both lists are sorted and deduplicated at build time.
  // Conjunct subsumption was tried here and is WRONG for splicing. "The query's
  // rows are a subset of this page's rows" is not enough: the splice puts cached
  // columns side by side with columns the reader is about to produce, and the
  // reader applies the FULL predicate. A page filtered by `A` beside a column
  // filtered by `A AND B` holds more rows than its neighbour -- measured as
  // "Column size mismatch: 548873 != 558097" on ClickBench q38. The two halves
  // must hold exactly the SAME rows, so an identical predicate is a requirement,
  // not conservatism. Reuse under a narrower predicate needs the cached rows
  // re-filtered before they can be spliced, which this path does not do.
  (void)required_filter_conjuncts;
  std::vector<std::shared_ptr<cudf::column>> out(column_names.size());
  if (row_groups.empty() || column_names.empty()) { return out; }

  std::lock_guard lock{_pages_mutex};
  auto const file_id = intern_lookup(file_path);
  if (file_id < 0) { return out; }
  for (std::size_t ci = 0; ci < column_names.size(); ++ci) {
    auto const column_id = intern_lookup(column_names[ci]);
    if (column_id < 0) { continue; }
    std::vector<cudf::column_view> parts;
    std::vector<std::shared_ptr<cudf::column>> keep_alive;
    bool complete = true;

    for (auto const row_group : row_groups) {
      // A row group is stored as a run of pages, page_index 0,1,2,... Walk it until
      // a page is missing; the run must be whole, because a gap in the middle would
      // silently drop rows from the column this hands back.
      std::size_t pages_found = 0;
      std::size_t pages_expected = 0;
      for (int page_index = 0;; ++page_index) {
        auto it = _pages.find(
          cached_page_key{file_id, column_id, static_cast<int>(row_group), page_index});
        if (it == _pages.end() || !it->second.data) { break; }
        // Every page of a run must agree on how many pages the run has. A row group can
        // be paged twice under different layouts -- a STRING column's rows-per-page comes
        // from the batch's average string length, which differs between batches -- and the
        // keys of the longer layout's tail do not collide with the shorter one's, so both
        // end up resident. Walking to the first miss then concatenates pages that overlap:
        // measured as a cached column of 11,122,762 rows against a row group of 10,000,000.
        if (pages_expected == 0) {
          pages_expected = it->second.pages_in_row_group;
        } else if (pages_expected != it->second.pages_in_row_group) {
          complete = false;
          break;
        }
        // The cached rows must have gone through the same predicate as the columns
        // about to be read beside them, or the two halves hold different rows. An
        // unfiltered page satisfies either case.
        if (!it->second.filter_signature.empty() &&
            it->second.filter_signature != required_filter_signature) {
          complete = false;
          break;
        }
        if (it->second.filter_signature.empty() && !required_filter_signature.empty()) {
          // Cache holds ALL rows, the reader is about to return only matching ones.
          complete = false;
          break;
        }
        it->second.last_access_tick = ++_page_tick;
        _lru.splice(_lru.end(), _lru, it->second.lru_it);  // most recently used
        parts.push_back(it->second.data->view());
        keep_alive.push_back(it->second.data);
        ++pages_found;
      }
      // Whole run or nothing: a short run means a page in the middle or at the end
      // was evicted, and serving it would drop those rows without a trace.
      if (!complete || pages_found == 0 || pages_found != pages_expected) {
        complete = false;
        break;
      }
    }

    if (!complete || parts.empty()) { continue; }
    if (parts.size() == 1) {
      out[ci] = std::move(keep_alive.front());
      continue;
    }
    out[ci] = std::shared_ptr<cudf::column>(
      cudf::concatenate(parts, stream, cudf::get_current_device_resource_ref()).release());
  }
  return out;
}

bool sirius_scan_manager::try_assign_cached_entries(op::scan::sirius_gpu_scan_operator* op,
                                                    bool* residual_out)
{
  // Every scan operator passes through here exactly once, so counting entries and
  // exits gives an exact hit ratio denominator. Without the miss log below, a scan
  // that simply finds no matching entry leaves no trace at all and the ratio can
  // only be bounded, not measured.
  SIRIUS_LOG_INFO("[fixed-page-cache] reuse_attempt operator='{}'", op->get_operator_id());
  const auto& table_info = op->get_ingestible().table_info();
  auto* parquet = dynamic_cast<op::scan::parquet_gpu_ingestible*>(&op->get_ingestible());
  if (parquet != nullptr && parquet->fixed_page_cache_has_dynamic_filters() &&
      !dynamic_filter_scan_cache_read_enabled()) {
    SIRIUS_LOG_INFO("[fixed-page-cache] reuse_skip reason=dynamic_filter_scan operator='{}'",
                    op->get_operator_id());
    return false;
  }

  // Why each candidate entry was rejected. reuse_miss used to log only that the
  // loop fell through, so a miss could be a wrong table, a filter that did not
  // match, or a provider that could not tile its pages -- three different bugs
  // with one symptom. 124 of 252 scans on SF100 end here; the counts say which.
  std::size_t miss_wrong_files     = 0;
  std::size_t miss_missing_columns = 0;
  std::size_t best_overlap         = 0;  // most of the request any same-file entry held
  std::size_t best_requested       = 0;  // columns the scan asked for
  std::size_t best_held            = 0;  // columns the widest same-file entry held
  std::size_t miss_no_columns      = 0;
  std::size_t miss_filter_mismatch = 0;
  std::size_t miss_no_provider     = 0;
  std::size_t candidates_seen      = 0;
  std::unordered_map<std::string, std::unordered_set<int>> surviving;
  bool surviving_computed = false;

  try {
    std::lock_guard pinned_entries_lock{_pinned_entries_mutex};
    for (auto const& [pinned_name, entry] : _pinned_entries) {
      ++candidates_seen;
      // Identity + serviceability gate: empty when this cache cannot serve the scan
      // (wrong format / file-set / table, or missing a requested column).
      if (entry.cache_info.can_serve_with_columns(table_info).empty()) {
        // Split the two causes the empty vector conflates: a cache for a
        // different table (the common case, since the loop walks every entry)
        // versus one for THIS table that simply lacks a column the scan reads.
        // Only the second is a cache-design problem.
        bool same_files = false;
        if (auto const* p =
              dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&table_info)) {
          auto these_files = entry.cache_info.resolved_file_paths;
          auto those_files = p->resolved_file_paths;
          std::sort(these_files.begin(), these_files.end());
          std::sort(those_files.begin(), those_files.end());
          same_files = these_files == those_files;
        }
        if (same_files) {
          ++miss_missing_columns;
          // How CLOSE the entry was, not just that it failed. The gather is
          // all-or-nothing across columns, so one absent column throws away every
          // column the entry does hold -- and whether serving those from cache and
          // the rest from parquet is worth building depends on that ratio, which
          // "missing_columns=72" alone cannot answer.
          std::unordered_set<std::size_t> held;
          held.reserve(entry.cache_info.column_ids.size());
          for (auto const& c : entry.cache_info.column_ids) {
            held.insert(static_cast<std::size_t>(c.GetPrimaryIndex()));
          }
          std::size_t overlap = 0;
          std::size_t requested = 0;
          if (auto const* p =
                dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&table_info)) {
            requested = p->column_ids.size();
            for (auto const& c : p->column_ids) {
              if (held.contains(static_cast<std::size_t>(c.GetPrimaryIndex()))) { ++overlap; }
            }
          }
          best_overlap     = std::max(best_overlap, overlap);
          best_requested   = requested;
          best_held        = std::max(best_held, entry.cache_info.column_ids.size());
        } else {
          ++miss_wrong_files;
        }
        continue;
      }
      // Refuse a short entry. num_rows is only a running total of what was
      // inserted, so an entry left behind by a mid-scan auto_cache_populate_failed
      // looks valid and silently serves fewer rows than the table has. Behaviour-
      // neutral while every entry is complete-or-erased; it becomes load-bearing
      // the moment partial residency makes short entries a normal state.
      if (!entry_is_complete(entry)) {
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] reuse_skip reason=incomplete_entry table='{}' rows={} "
          "table_total_rows={}",
          pinned_name,
          entry.num_rows,
          entry.cache_info.table_total_rows);
        continue;
      }
      // Serve cached columns in the ingestible's materialized (disk-decode) order rather
      // than raw column_ids order, so post_filter_and_project's index-based filter and
      // projection bind to the same columns they would on the disk read path.
      auto cols = gather_by_primary_index(entry.cache_info.column_ids,
                                          op->get_ingestible().materialized_column_order());
      if (cols.empty()) {  // defensive: materialized set must be a cache subset
        ++miss_no_columns;
        continue;
      }
      if (!entry.cache_info.filter_signature.empty()) {
        // Byte-identical predicate is the cheap path and the common one inside a
        // benchmark that repeats its query set. Outside one, predicates differ by
        // a constant -- a different date window, a different id -- and the text
        // never matches even though the cached rows are a superset of what the new
        // query wants. So fall back to asking whether this scan's predicate is
        // narrower than the one that filled the entry.
        bool reusable = parquet != nullptr && parquet->fixed_page_cache_filter_signature() ==
                                                entry.cache_info.filter_signature;
        if (!reusable && parquet != nullptr) {
          bool consumer_analyzable = false;
          auto const consumer_ranges =
            parquet->fixed_page_cache_filter_ranges(consumer_analyzable, /*strict=*/false);
          reusable                   = filter_ranges_subsume(entry.cache_info.filter_ranges,
                                        entry.cache_info.filter_analyzable,
                                        consumer_ranges,
                                        consumer_analyzable);
          if (reusable) {
            SIRIUS_LOG_INFO(
              "[fixed-page-cache] filter_subsumed table='{}' operator='{}' producer='{}'",
              pinned_name,
              op->get_operator_id(),
              entry.cache_info.filter_signature);
          }
        }
        if (!reusable) {
          ++miss_filter_mismatch;
          continue;
        }
      }
      // Row groups the scan's static predicate cannot rule out. Computed once,
      // lazily, and only when an entry is otherwise serviceable -- it parses no
      // footer that the scan has not already parked.
      if (parquet != nullptr && !surviving_computed) {
        surviving = parquet->surviving_row_groups(
          [this](std::string_view path) -> std::shared_ptr<io::sirius_ioctx> {
            return ioctx_for_path(path);
          });
        surviving_computed = true;
      }
      // With the predicate out of the cache key, an entry can hold chunks filtered
      // differently, so each chunk is cleared against THIS scan's predicate instead.
      bool consumer_analyzable = false;
      std::vector<cache_filter_range> consumer_ranges;
      bool const keyless = entry.cache_info.filter_signature.empty() &&
                           entry.cache_info.chunks_carry_filter_ranges;
      if (keyless && parquet != nullptr) {
        consumer_ranges = parquet->fixed_page_cache_filter_ranges(consumer_analyzable,
                                                                  /*strict=*/false);
      }
      auto provider = make_provider_for_pinned_entry(entry,
                                                     cols,
                                                     &_shared_dictionaries,
                                                     surviving.empty() ? nullptr : &surviving,
                                                     keyless ? &consumer_ranges : nullptr,
                                                     consumer_analyzable,
                                                     keyless && parquet != nullptr
                                                       ? parquet->fixed_page_cache_filter_signature()
                                                       : std::string{});
      if (!provider) {
        // usable() is false: no page range survived build_ranges, i.e. no chunk is
        // covered by every selected column at page-aligned boundaries.
        ++miss_no_provider;
        continue;
      }

      // Partial coverage: tell the ingestible which row groups the cache serves so
      // it reads only the complement, and mark the pipeline so both sources feed
      // the connector. The row groups come from the entry's recorded provenance --
      // never from arithmetic on chunk_index, which is an arrival counter.
      // An entry that does not hold the whole table needs a residual even when the
      // provider covers every row the entry HAS -- covers_entire_entry() is about
      // the entry, not the table, so an entry caught at 10,000,000 of 99,997,497
      // rows reads as fully covered and would hand the scan a tenth of the table.
      bool const whole_table = entry_holds_whole_table(entry);
      bool residual          = false;
      if (auto const* page_provider =
            dynamic_cast<fixed_page_databatch_provider const*>(provider.get());
          page_provider != nullptr && parquet != nullptr &&
          (!page_provider->covers_entire_entry() || !whole_table)) {
        auto cached_groups = page_provider->covered_row_groups();
        if (cached_groups.empty()) { continue; }  // nothing nameable: take the miss
        std::size_t cached_row_group_count = 0;
        for (auto const& [path, groups] : cached_groups) { cached_row_group_count += groups.size(); }
        parquet->set_cached_row_groups(std::move(cached_groups));
        residual = true;
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] partial_reuse table='{}' cached_row_groups={} across {} chunk(s) of "
          "{} operator='{}'",
          pinned_name,
          cached_row_group_count,
          page_provider->covered_chunks().size(),
          entry.chunk_provenance_by_index.size(),
          op->get_operator_id());
      }
      if (!whole_table && !residual) {
        // Partial entry whose complement cannot be named (no parquet reader, or no
        // row group survived): serving it would silently drop the rest.
        SIRIUS_LOG_INFO(
          "[fixed-page-cache] reuse_skip reason=unnameable_residual table='{}' operator='{}'",
          pinned_name,
          op->get_operator_id());
        continue;
      }
      _metadata_processor->use_cached_entries_for_pipeline(op, std::move(provider), residual);
      if (residual_out != nullptr) { *residual_out = residual; }
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
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] reuse_miss operator='{}' candidates={} wrong_files={} "
    "missing_columns={} no_columns={} filter_mismatch={} no_provider={} "
    "requested_columns={} best_overlap={} best_entry_columns={}",
    op->get_operator_id(),
    candidates_seen,
    miss_wrong_files,
    miss_missing_columns,
    miss_no_columns,
    miss_filter_mismatch,
    miss_no_provider,
    best_requested,
    best_overlap,
    best_held);
  return false;
}

}  // namespace sirius::scan_manager
