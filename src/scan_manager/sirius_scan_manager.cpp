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

#include "exec/thread_pool.hpp"
#include "io/gpu_ingestible.hpp"
#include "io/parquet_helpers.hpp"
#include "io/prefetching_cache.hpp"
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan_manager/parquet_metadata.hpp"
#include "scan_manager/round_robin_strategy.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
// wdy start
#include <cstdlib>
// wdy end
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
// wdy start
#include <string>
// wdy end
#include <utility>

namespace sirius::scan_manager {

namespace {

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
  return budget;
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

  std::size_t evicted_pages = 0;
  std::size_t evicted_bytes = 0;
  for (auto& [_, col_pages] : entry.fixed_width_pages_by_column) {
    for (auto& page : col_pages) {
      if (page.state != fixed_width_page_state::resident || page.key.device_id < 0) { continue; }
      auto& device_bytes = resident_bytes_by_device[page.key.device_id];
      if (device_bytes <= budget) { continue; }
      page.state = fixed_width_page_state::evicted;
      page.owned_column.reset();
      device_bytes -= std::min(device_bytes, page.num_bytes);
      ++entry.fixed_width_page_metrics.eviction_count;
      ++evicted_pages;
      evicted_bytes += page.num_bytes;
    }
  }

  if (evicted_pages != 0) {
    SIRIUS_LOG_INFO(
      "[fixed-page-cache] page_budget applied table='{}' budget_bytes_per_gpu={} "
      "evicted_pages={} evicted_bytes={}",
      name,
      budget,
      evicted_pages,
      evicted_bytes);
  }
}

// wdy end
}  // namespace

sirius_scan_manager::sirius_scan_manager(
  scan_manager_config config, cucascade::memory::fixed_size_host_memory_resource* host_fsmr)
  : _config(std::move(config)),
    _thread_pool(_config.thread_pool.num_threads,
                 _config.thread_pool.thread_name_prefix,
                 _config.thread_pool.cpu_affinity_list),
    _dispatcher(
      std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads)),
    _factory(_pinned_entries)
{
  // The local uring backend is constructed unconditionally: the DuckDB-native
  // GPU scan needs it for host reads even when use_sirius_datasource=false.
  // That flag instead gates local-path CLAIMING in create_datasource (local
  // parquet falls back to cudf/KvikIO when false).
  auto ioctx = std::make_shared<sirius::io::uring_ioctx>(
    /*host_ring_depth=*/16u,
    /*ring_entries=*/_config.uring_ring_entries,
    /*n_reactors=*/_config.uring_n_reactors,
    /*bounce_slot_size=*/sirius::io::CHUNK_SIZE);
  _io_ctxs.push_back(std::move(ioctx));

  if (_config.s3_config) {
    auto s3_cfg                 = *_config.s3_config;
    s3_cfg.host_memory_resource = host_fsmr;
    std::shared_ptr<sirius::io::sirius_ioctx> s3_backend;
    if (_config.s3_use_async_backend) {
      // Async backend (default): the libcurl-multi reactor owns its own worker
      // thread, so no s3_thread_pool is created. Retry knobs are forwarded so
      // the async backend is config-equivalent to the blocking one.
      s3_backend = std::make_shared<sirius::io::s3::s3_ioctx>(std::move(s3_cfg.creds),
                                                              s3_cfg.request_timeout_s,
                                                              s3_cfg.ca_bundle_path,
                                                              s3_cfg.tls_verify,
                                                              s3_cfg.max_connections,
                                                              host_fsmr,
                                                              s3_cfg.max_retry_attempts,
                                                              s3_cfg.retry_backoff_base,
                                                              s3_cfg.retry_jitter,
                                                              s3_cfg.honor_retry_after);
    } else {
      // Blocking backend (fallback): fan async work out over a dedicated pool.
      if (s3_cfg.async_thread_pool == nullptr) {
        _s3_thread_pool = std::make_unique<sirius::exec::static_thread_pool>(
          _config.s3_thread_pool.num_threads,
          _config.s3_thread_pool.thread_name_prefix,
          _config.s3_thread_pool.cpu_affinity_list);
        s3_cfg.async_thread_pool = _s3_thread_pool.get();
      }
      s3_backend = std::make_shared<sirius::io::s3::s3_blocking_ioctx>(std::move(s3_cfg));
    }
    _io_ctxs.push_back(std::move(s3_backend));
  }

  if (_config.enable_prefetch_cache && host_fsmr != nullptr) {
    auto const slab_bytes = host_fsmr->get_block_size() *
                            static_cast<std::size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB);
    auto const max_slabs =
      static_cast<uint32_t>((_config.prefetch_buffer_pool_bytes + slab_bytes - 1) / slab_bytes);
    _prefetch_buffer_pool = std::make_unique<sirius::io::buffer_pool>(*host_fsmr, max_slabs);
    for (auto& ctx : _io_ctxs) {
      if (ctx) {
        ctx->initialize_cache(*_prefetch_buffer_pool, _config.prefetch_inflight_budget_chunks);
      }
    }
  }

  SIRIUS_LOG_DEBUG("[sirius_scan_manager] constructed with {} IO backend(s)", _io_ctxs.size());
}

void clear_fixed_width_filter_mask_cache(pinned_entry& entry)
{
  if (!entry.fixed_width_filter_mask_cache_mutex) {
    entry.fixed_width_filter_mask_cache.clear();
    return;
  }
  std::lock_guard<std::mutex> guard(*entry.fixed_width_filter_mask_cache_mutex);
  for (auto& [_, mask_entry] : entry.fixed_width_filter_mask_cache) {
    if (!mask_entry.mask) { continue; }
    if (mask_entry.device_id >= 0) {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{mask_entry.device_id}};
      mask_entry.mask.reset();
    } else {
      mask_entry.mask.reset();
    }
  }
  entry.fixed_width_filter_mask_cache.clear();
}

void clear_fixed_width_filter_mask_caches(
  std::unordered_map<std::string, pinned_entry>& pinned_entries)
{
  for (auto& [_, entry] : pinned_entries) {
    clear_fixed_width_filter_mask_cache(entry);
  }
}

sirius_scan_manager::~sirius_scan_manager()
{
  clear_fixed_width_filter_mask_caches(_pinned_entries);
  stop();
  _io_ctxs.clear();
  if (_s3_thread_pool) { _s3_thread_pool->stop(); }
  _s3_thread_pool.reset();
  _prefetch_buffer_pool.reset();
}

namespace {

// Walk @c ioctxs and return the first @c ctx whose @c supports(path) is
// true, or nullptr. Also tries with @c "file://" prefix stripped because
// @c uring_reactor::supports (from #740) calls @c is_regular_file on the
// raw input — so it accepts bare absolute paths but not @c file:// URIs.
// Stripping at the dispatch layer keeps #740's code untouched and works
// for the both-shape inputs Sirius's parquet plans can produce.
template <typename Container, typename Out>
Out lookup_supporting(Container const& ioctxs,
                      std::string_view path,
                      Out (*get_value)(typename Container::value_type const&))
{
  for (auto const& ctx : ioctxs) {
    if (ctx && ctx->supports(path)) return get_value(ctx);
  }
  constexpr std::string_view kFileScheme = "file://";
  if (path.size() > kFileScheme.size() && path.substr(0, kFileScheme.size()) == kFileScheme) {
    auto bare = path.substr(kFileScheme.size());
    for (auto const& ctx : ioctxs) {
      if (ctx && ctx->supports(bare)) return get_value(ctx);
    }
  }
  return Out{};
}

}  // namespace

parquet_bind_result sirius_scan_manager::describe_parquet(std::string const& uri)
{
  auto datasource = create_datasource(uri);
  if (!datasource) {
    throw std::runtime_error("[sirius_scan_manager::describe_parquet] no backend supports URI: " +
                             uri);
  }

  // Footer-only fetch + Thrift parse — the same path parquet_split_provider's
  // run_batch takes on a metadata-cache miss, so bind and scan agree on how
  // the footer is read.
  auto footer_buffer         = cudf::io::parquet::fetch_footer_to_host(*datasource);
  auto const footer_byte_len = footer_buffer->size();
  auto reader_options        = cudf::io::parquet_reader_options::builder().build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()),
    reader_options};
  auto file_metadata         = reader.parquet_metadata();
  auto const footer_num_rows = file_metadata.num_rows;

  auto schema = sirius::io::parquet_helpers::extract_schema(file_metadata);

  // Footer-parse reuse: a metadata-only insert (empty ranges => no chunk
  // prefetch) lets the subsequent scan's get_metadata hit, so the footer is
  // Thrift-parsed once instead of twice.
  if (auto* cache = datasource->io_ctx()->cache(); cache != nullptr) {
    auto metadata = std::make_shared<parquet_metadata>(
      std::make_shared<cudf::io::parquet::FileMetaData const>(std::move(file_metadata)),
      footer_byte_len);
    cache->insert(*datasource->io_object(), std::move(metadata), /*ranges=*/{});
  }

  parquet_bind_result result;
  result.return_types   = std::move(schema.types);
  result.names          = std::move(schema.names);
  result.object_size    = datasource->size();
  result.total_num_rows = static_cast<std::size_t>(footer_num_rows);
  return result;
}

void sirius_scan_manager::prepare_for_query(
  const sirius::planner::query& query,
  std::unordered_map<int, cucascade::memory::memory_space*> const& gpu_memory_spaces)
{
  reset();

  for (auto const& ctx : _io_ctxs) {
    if (ctx && ctx->cache()) { ctx->cache()->refresh_cache(); }
  }

  SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] pipelines={} gpu_memory_spaces={}",
                   query.get_pipelines().size(),
                   gpu_memory_spaces.size());

  // Device placement for fresh-read scan splits. Snapshot the query's GPU set
  // in a stable (sorted) order and build one round-robin strategy shared across
  // every provider, so the walk spreads splits evenly over all GPUs across the
  // whole scan stage instead of restarting per scan operator. A provider stamps
  // the chosen device onto its splits' operating data; the task creator reads
  // it back when building the pipeline task.
  std::vector<int> device_ids;
  device_ids.reserve(gpu_memory_spaces.size());
  for (auto const& [device_id, space] : gpu_memory_spaces) {
    device_ids.push_back(device_id);
  }
  std::sort(device_ids.begin(), device_ids.end());
  auto round_robin = std::make_shared<round_robin_strategy>(std::move(device_ids));

  for (auto const& pipeline : query.get_pipelines()) {
    if (!pipeline) { continue; }
    auto source = pipeline->get_source();
    if (!source) { continue; }
    if (source->type != ::sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }

    auto* op = &source->Cast<op::scan::sirius_gpu_scan_operator>();
    if (_providers_by_op.find(op) != _providers_by_op.end()) { continue; }

    auto table_info = op->take_table_info();
    if (!table_info) { continue; }

    auto ingestible =
      _factory.produce(std::move(table_info), *this, gpu_memory_spaces, op->get_operator_id());
    if (!ingestible) { continue; }
    op->install_ingestible(std::move(ingestible));

    // Operator is now the sole shared_ptr owner; provider borrows the
    // ingestible by reference. enable_shared_from_this lets any consumer
    // promote to shared_ptr on demand.
    auto provider = std::make_unique<split_provider>(op->get_ingestible());
    // Spread fresh-read splits across GPUs for every scan source — parquet and
    // duckdb-native alike both emit non-resident scan_operator_input splits.
    // Installing the strategy on every provider is safe: resident pinned-cache
    // splits are left untouched by split_provider::apply_balancing's
    // is_resident() guard, so their data-locality placement is preserved.
    provider->set_balancing_strategy(round_robin, pipeline->get_pipeline_id());
    op->set_split_connector(std::make_unique<split_connector>());
    _providers_by_op.emplace(op, std::move(provider));
    _scan_op_order.push_back(op);

    SIRIUS_LOG_DEBUG("[sirius_scan_manager::prepare_for_query] registered gpu scan op_id={}",
                     op->get_operator_id());
  }

  if (_scan_op_order.empty()) { return; }

  start_metadata_processing();
}

void sirius_scan_manager::start_metadata_processing()
{
  for (auto* op : _scan_op_order) {
    auto it = _providers_by_op.find(op);
    if (it == _providers_by_op.end()) { continue; }
    auto* connector = op->get_split_connector();
    if (connector == nullptr) { continue; }

    try {
      // run() is fire-and-forget: it enqueues workers and returns immediately.
      // Worker exceptions ride on connector.close(exception_ptr) and surface
      // when the consumer drains via get_next_split().
      it->second->run(*_dispatcher, *connector);
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("[sirius_scan_manager] driver: provider failed to start: {}", e.what());
      // Synchronous failure inside run() (e.g. scheduler.enqueue throwing)
      // bypasses the worker error path, so forward it through the connector
      // here. close() is idempotent and keeps the first stored exception.
      connector->close(std::current_exception());
    }
  }
}

void sirius_scan_manager::reset()
{
  _dispatcher->request_stop();
  _dispatcher->wait_for_all();
  _scan_op_order.clear();
  _providers_by_op.clear();
  _dispatcher =
    std::make_unique<exec::scoped_dispatcher>(_thread_pool, _config.thread_pool.num_threads);
}

void sirius_scan_manager::start() {}

void sirius_scan_manager::stop()
{
  reset();
  // Since the scan-manager cleanup (#913) the scan_manager OWNS the IO
  // backends, the S3 thread pool and the prefetch buffer pool; they are torn
  // down in the destructor (_io_ctxs.clear() -> S3 pool stop -> buffer pool
  // reset). stop() only halts the scan-orchestration pool so it stays safe to
  // call while backends may still serve in-flight reads.
  _thread_pool.stop();
}

void sirius_scan_manager::insert_pinned_entry(
  const std::string& name,
  std::vector<std::string> column_names,
  std::vector<std::string> file_paths,
  std::vector<std::unique_ptr<cudf::table>> data_tables,
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
  bool is_partial)
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

  auto existing_it = _pinned_entries.find(name);
  if (existing_it != _pinned_entries.end()) {
    // Same-row-count merge only applies when the completeness contracts match.
    // Mixing a full pin with a partial pin produces an entry whose columns came
    // from different row coverage — drop and rebuild instead.
    if (existing_it->second.num_rows == new_num_rows &&
        existing_it->second.is_partial == is_partial) {
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
      // wdy start
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
          auto& chunks           = entry.data_batches_by_column[column_names[i]];
          auto const chunk_index = chunks.size();
          auto* chunk_space      = chunk_index < entry.chunk_memory_spaces.size()
                                     ? entry.chunk_memory_spaces[chunk_index]
                                     : nullptr;
          auto const page_file_path =
            entry.file_paths.empty() ? std::string{} : entry.file_paths.front();
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
                                         own_page_storage);
          if (own_page_storage) {
            chunks.emplace_back(nullptr);
          } else {
            chunks.emplace_back(std::move(column));
          }
        }
      }
      // Append any new column names to the entry's column_names list so its
      // metadata reflects the union of pinned columns.
      for (auto& cn : column_names) {
        if (std::find(entry.column_names.begin(), entry.column_names.end(), cn) ==
            entry.column_names.end()) {
          entry.column_names.push_back(std::move(cn));
        }
      }
      entry.fixed_width_page_metrics = {};
      apply_fixed_width_page_budget(entry, name);
      auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
      entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
      entry.fixed_width_page_metrics.eviction_count = eviction_count;
      SIRIUS_LOG_INFO(
        "[fixed-page-cache] page_directory indexed table='{}' fixed_cols={} pages={} "
        "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} "
        "eviction_count={} directory_entries={} page_bytes={} rows={} partial={} owned_pages={}",
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
        entry.is_partial,
        own_page_storage ? 1 : 0);
      // wdy end
      return;
    }
    // Row count or completeness contract differs → drop the stale entry and rebuild below.
    _pinned_entries.erase(existing_it);
  }

  pinned_entry entry;
  entry.column_names                = std::move(column_names);
  entry.file_paths                  = std::move(file_paths);
  entry.chunk_memory_spaces         = std::move(chunk_memory_spaces);
  entry.tier                        = cucascade::memory::Tier::GPU;
  entry.num_rows                    = new_num_rows;
  entry.is_partial                  = is_partial;
  // wdy start
  entry.fixed_width_page_size_bytes = fixed_width_page_size_bytes();
  bool const own_page_storage       = fixed_page_owned_pages_enabled() && is_auto_fixed_page_entry(name);

  for (auto& table : data_tables) {
    if (!table) { continue; }
    auto cols = table->release();
    if (cols.size() != entry.column_names.size()) {
      throw std::runtime_error("[sirius_scan_manager::insert_pinned_entry] table column count " +
                               std::to_string(cols.size()) + " does not match column_names size " +
                               std::to_string(entry.column_names.size()));
    }
    for (std::size_t i = 0; i < cols.size(); ++i) {
      auto column            = std::move(cols[i]);
      auto& chunks           = entry.data_batches_by_column[entry.column_names[i]];
      auto const chunk_index = chunks.size();
      auto* chunk_space      = chunk_index < entry.chunk_memory_spaces.size()
                                 ? entry.chunk_memory_spaces[chunk_index]
                                 : nullptr;
      auto const page_file_path =
        entry.file_paths.empty() ? std::string{} : entry.file_paths.front();
      std::size_t chunk_global_row_offset = 0;
      if (own_page_storage) {
        auto pages_it = entry.fixed_width_pages_by_column.find(entry.column_names[i]);
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
                                     entry.column_names[i],
                                     *column,
                                     chunk_index,
                                     chunk_global_row_offset,
                                     chunk_space,
                                     entry.fixed_width_page_size_bytes,
                                     own_page_storage);
      if (own_page_storage) {
        chunks.emplace_back(nullptr);
      } else {
        chunks.emplace_back(std::move(column));
      }
    }
  }

  entry.fixed_width_page_metrics = {};
  apply_fixed_width_page_budget(entry, name);
  auto eviction_count = entry.fixed_width_page_metrics.eviction_count;
  entry.fixed_width_page_metrics = compute_fixed_width_page_directory_metrics(entry);
  entry.fixed_width_page_metrics.eviction_count = eviction_count;
  SIRIUS_LOG_INFO(
    "[fixed-page-cache] page_directory indexed table='{}' fixed_cols={} pages={} "
    "resident_pages={} resident_bytes={} stats_pages={} evicted_pages={} eviction_count={} "
    "directory_entries={} page_bytes={} rows={} partial={} owned_pages={}",
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
    entry.is_partial,
    own_page_storage ? 1 : 0);
  // wdy end

  _pinned_entries[name] = std::move(entry);
}

void sirius_scan_manager::insert_pinned_entry_host(
  const std::string& name,
  std::vector<std::string> column_names,
  std::vector<std::string> file_paths,
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
  cucascade::memory::memory_space& memory_space,
  bool is_partial)
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
  entry.column_names = std::move(column_names);
  entry.file_paths   = std::move(file_paths);
  entry.tier         = cucascade::memory::Tier::HOST;
  entry.memory_space = &memory_space;
  entry.num_rows     = new_num_rows;
  entry.host_chunks  = std::move(host_chunks);
  entry.is_partial   = is_partial;

  _pinned_entries[name] = std::move(entry);
}

std::shared_ptr<sirius::io::sirius_datasource> sirius_scan_manager::create_datasource(
  std::string_view path) const
{
  auto file_path = normalize_path(std::string(path));
  // use_sirius_datasource=false keeps LOCAL paths on cudf's bundled datasource
  // (KvikIO fallback): they are deliberately left unclaimed so slices carry a
  // null datasource and reads fall back to cudf::io::datasource::create.
  // Object-store paths (s3:// etc.) always resolve through their backend —
  // this flag does not disable S3.
  if (!_config.use_sirius_datasource && file_path.find("://") == std::string::npos) {
    return nullptr;
  }
  for (auto const& ctx : _io_ctxs) {
    if (ctx && ctx->supports(file_path)) {
      auto io_object = ctx->create_io_object(file_path.data());
      return ctx->make_datasource(io_object);
    }
  }
  return nullptr;
}

void sirius_scan_manager::remove_pinned_entry(const std::string& name)
{
  _pinned_entries.erase(name);
}

}  // namespace sirius::scan_manager
