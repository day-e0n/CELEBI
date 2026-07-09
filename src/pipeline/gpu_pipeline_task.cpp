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

#include "pipeline/gpu_pipeline_task.hpp"

#include "cudf/cudf_utils.hpp"
#include "log/logging.hpp"
#include "memory/defragmenter_oom_policy.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "telemetry/telemetry_context.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"

#include <nvtx3/nvtx3.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <data/data_batch_utils.hpp>

#include <cstdint>
#include <cstdlib>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace sirius {
namespace pipeline {

namespace {

void validate_operator_output_types(const op::operator_data* data,
                                    const op::sirius_physical_operator& op)
{
  if (data == nullptr) { return; }
  auto* pipelineable_data = dynamic_cast<const op::pipelineable_operator_data*>(data);
  if (pipelineable_data == nullptr) { return; }
  const auto& expected_types = op.get_types();
  const auto& batches        = pipelineable_data->get_data_batches();
  for (size_t batch_index = 0; batch_index < batches.size(); batch_index++) {
    const auto& batch = batches[batch_index];
    if (!batch) { continue; }
    cudf::table_view tbl = get_cudf_table_view(*batch);
    if (static_cast<size_t>(tbl.num_columns()) != expected_types.size()) {
      // bobbi (todo): delim join will return this warning for now, but there is no bug here, so we
      // can ignore it. we can do something about this after gtc
      SIRIUS_LOG_WARN(
        "gpu_pipeline_task: operator '{}' (id={}) output batch {} column count mismatch: got "
        "{}, expected {}",
        op.get_name(),
        op.get_operator_id(),
        batch_index,
        tbl.num_columns(),
        expected_types.size());
      return;
    }
    for (cudf::size_type c = 0; c < tbl.num_columns(); c++) {
      cudf::data_type expected_cudf = sirius::get_cudf_type(expected_types[c]);
      cudf::data_type actual        = tbl.column(c).type();
      if (actual != expected_cudf) {
        SIRIUS_LOG_WARN(
          "gpu_pipeline_task: operator '{}' (id={}) output batch {} column {} datatype "
          "mismatch: got {}, expected {}",
          op.get_name(),
          op.get_operator_id(),
          batch_index,
          c,
          cudf::type_to_name(actual),
          cudf::type_to_name(expected_cudf));
        return;
      }
    }
  }
}

// Authoritative source for the GPU id used by per-task log lines: the
// executor's _per_thread_init runs cudaSetDevice(executor_gpu) on every
// worker thread, and compute_task wraps the per-task work in
// rmm::cuda_set_device_raii on the same id, so cudaGetDevice here reflects
// the executor that is running this task.
int current_gpu_id()
{
  int dev = -1;
  (void)::cudaGetDevice(&dev);
  return dev;
}

// wdy start
// locality audit: for each input batch, record the locality of the batch relative to the target memory space (if any) and accumulate the bytes in each category (local, remote GPU, host, disk). This is useful for understanding data movement and locality in GPU pipelines.
// for breakdown
const char* tier_name(cucascade::memory::Tier tier) 
{
  switch (tier) {
    case cucascade::memory::Tier::GPU: return "GPU";
    case cucascade::memory::Tier::HOST: return "HOST";
    case cucascade::memory::Tier::DISK: return "DISK";
    default: return "UNKNOWN";
  }
}

struct locality_bytes_snapshot {
  size_t input_bytes      = 0;
  size_t local_bytes      = 0;
  size_t remote_gpu_bytes = 0;
  size_t host_bytes       = 0;
  size_t disk_bytes       = 0;
  size_t batch_count      = 0;
};

// Adds the given batch's bytes to the snapshot, categorizing them based on their locality relative to the target memory space.
void add_batch_to_snapshot(locality_bytes_snapshot& snapshot,
                           const cucascade::memory::memory_space* space,
                           size_t bytes,
                           const cucascade::memory::memory_space* target_space)
{
  snapshot.input_bytes += bytes;
  snapshot.batch_count++;
  if (space == nullptr) { return; }
  if (target_space != nullptr && space->get_id() == target_space->get_id()) {
    snapshot.local_bytes += bytes;
    return;
  }
  switch (space->get_tier()) {
    case cucascade::memory::Tier::GPU: snapshot.remote_gpu_bytes += bytes; break;
    case cucascade::memory::Tier::HOST: snapshot.host_bytes += bytes; break;
    case cucascade::memory::Tier::DISK: snapshot.disk_bytes += bytes; break;
    default: break;
  }
}

// Summarizes the locality of unlocked batches in the given operator data relative to the target memory space.
locality_bytes_snapshot summarize_unlocked_batches(
  const op::pipelineable_operator_data& data, const cucascade::memory::memory_space* target_space)
{
  locality_bytes_snapshot snapshot;
  for (const auto& batch : data.get_data_batches()) {
    if (!batch) { continue; }
    auto ro = batch->to_read_only();
    if (!ro.get_data()) { continue; }
    add_batch_to_snapshot(
      snapshot, ro.get_memory_space(), ro.get_data()->get_size_in_bytes(), target_space);
  }
  return snapshot;
}

// Summarizes the locality of locked batches in the given operator data relative to the target memory space.
locality_bytes_snapshot summarize_locked_batches(
  const op::pipelineable_operator_data& data, const cucascade::memory::memory_space* target_space)
{
  locality_bytes_snapshot snapshot;
  for (const auto& ro : data.get_read_only_batches(false)) {
    if (!ro.get_data()) { continue; }
    add_batch_to_snapshot(
      snapshot, ro.get_memory_space(), ro.get_data()->get_size_in_bytes(), target_space);
  }
  return snapshot;
}

// Logs a snapshot of the locality of input batches relative to the target memory space, including counts of bytes in each category and the number of batches.
void log_locality_snapshot(const char* phase,
                           const sirius_pipeline* pipeline,
                           uint64_t task_id,
                           std::optional<int> preferred_device_id,
                           int actual_gpu,
                           const cucascade::memory::memory_space* target_space,
                           const locality_bytes_snapshot& snapshot)
{
  SIRIUS_LOG_INFO(
    "[locality-audit] {} pipeline_id={} task_id={} preferred_device={} actual_gpu={} "
    "target_tier={} target_device={} input_bytes={} local_bytes={} remote_gpu_bytes={} "
    "host_bytes={} disk_bytes={} batch_count={}",
    phase,
    pipeline ? pipeline->get_pipeline_id() : 0,
    task_id,
    preferred_device_id.value_or(-1),
    actual_gpu,
    target_space ? tier_name(target_space->get_tier()) : "NONE",
    target_space ? target_space->get_device_id() : -1,
    snapshot.input_bytes,
    snapshot.local_bytes,
    snapshot.remote_gpu_bytes,
    snapshot.host_bytes,
    snapshot.disk_bytes,
    snapshot.batch_count);
}
// wdy end

// wdy start
// Returns a string representation of the operator data type for logging purposes.
const char* operator_data_type_name(op::operator_data_type type)
{
  switch (type) {
    case op::operator_data_type::BASE: return "BASE";
    case op::operator_data_type::PIPELINEABLE: return "PIPELINEABLE";
    case op::operator_data_type::PARTITIONED: return "PARTITIONED";
    case op::operator_data_type::GPU_SCAN: return "GPU_SCAN";
    default: return "UNKNOWN";
  }
}

// to be used in log_stage_audit to log the stage kind (SCAN, FILTER, PROJECTION, JOIN, AGGREGATE, SORT, LIMIT, PARTITION, CONCAT, CTE, RESULT, OTHER) based on the operator type.
const char* stage_kind(op::SiriusPhysicalOperatorType type)
{
  switch (type) {
    case op::SiriusPhysicalOperatorType::TABLE_SCAN:
    case op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN:
    case op::SiriusPhysicalOperatorType::CHUNK_SCAN:
    case op::SiriusPhysicalOperatorType::RECURSIVE_CTE_SCAN:
    case op::SiriusPhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN:
    case op::SiriusPhysicalOperatorType::CTE_SCAN:
    case op::SiriusPhysicalOperatorType::DELIM_SCAN:
    case op::SiriusPhysicalOperatorType::EXPRESSION_SCAN:
    case op::SiriusPhysicalOperatorType::POSITIONAL_SCAN:
    case op::SiriusPhysicalOperatorType::DUCKDB_SCAN:
    case op::SiriusPhysicalOperatorType::PARQUET_SCAN:
    case op::SiriusPhysicalOperatorType::ICEBERG_SCAN:
    case op::SiriusPhysicalOperatorType::CPU_SOURCE:
    case op::SiriusPhysicalOperatorType::GPU_SCAN: return "SCAN";
    case op::SiriusPhysicalOperatorType::FILTER: return "FILTER";
    case op::SiriusPhysicalOperatorType::PROJECTION: return "PROJECTION";
    case op::SiriusPhysicalOperatorType::HASH_JOIN:
    case op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN:
    case op::SiriusPhysicalOperatorType::BLOCKWISE_NL_JOIN:
    case op::SiriusPhysicalOperatorType::CROSS_PRODUCT:
    case op::SiriusPhysicalOperatorType::PIECEWISE_MERGE_JOIN:
    case op::SiriusPhysicalOperatorType::IE_JOIN:
    case op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN:
    case op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN:
    case op::SiriusPhysicalOperatorType::POSITIONAL_JOIN:
    case op::SiriusPhysicalOperatorType::ASOF_JOIN: return "JOIN";
    case op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
    case op::SiriusPhysicalOperatorType::HASH_GROUP_BY:
    case op::SiriusPhysicalOperatorType::PERFECT_HASH_GROUP_BY:
    case op::SiriusPhysicalOperatorType::PARTITIONED_AGGREGATE:
    case op::SiriusPhysicalOperatorType::MERGE_GROUP_BY:
    case op::SiriusPhysicalOperatorType::MERGE_AGGREGATE: return "AGGREGATE";
    case op::SiriusPhysicalOperatorType::ORDER_BY:
    case op::SiriusPhysicalOperatorType::MERGE_SORT:
    case op::SiriusPhysicalOperatorType::SORT_PARTITION:
    case op::SiriusPhysicalOperatorType::SORT_SAMPLE: return "SORT";
    case op::SiriusPhysicalOperatorType::LIMIT:
    case op::SiriusPhysicalOperatorType::STREAMING_LIMIT:
    case op::SiriusPhysicalOperatorType::LIMIT_PERCENT:
    case op::SiriusPhysicalOperatorType::TOP_N:
    case op::SiriusPhysicalOperatorType::MERGE_TOP_N: return "LIMIT";
    case op::SiriusPhysicalOperatorType::PARTITION: return "PARTITION";
    case op::SiriusPhysicalOperatorType::CONCAT:
    case op::SiriusPhysicalOperatorType::UNION: return "CONCAT";
    case op::SiriusPhysicalOperatorType::CTE:
    case op::SiriusPhysicalOperatorType::RECURSIVE_CTE:
    case op::SiriusPhysicalOperatorType::RECURSIVE_KEY_CTE: return "CTE";
    case op::SiriusPhysicalOperatorType::RESULT_COLLECTOR: return "RESULT";
    default: return "OTHER";
  }
}

// Summarizes the number of batches, rows, columns, and total bytes in operator data.
struct stage_data_summary {
  size_t batches = 0;
  size_t rows    = 0;
  size_t columns = 0;
  size_t bytes   = 0;
};

// Summarizes the number of batches, rows, columns, and total bytes in the given operator data.
stage_data_summary summarize_stage_data(const op::operator_data& data)
{
  stage_data_summary summary;
  auto* p_data = dynamic_cast<const op::pipelineable_operator_data*>(&data);
  if (p_data == nullptr) { return summary; }

  for (auto const& batch : p_data->get_read_only_batches(false)) {
    if (!batch.get_data()) { continue; }
    auto view = get_cudf_table_view(batch);
    summary.batches++;
    summary.rows += static_cast<size_t>(view.num_rows());
    summary.columns += static_cast<size_t>(view.num_columns());
    summary.bytes += batch.get_data()->get_size_in_bytes();
  }
  return summary;
}
// log_stage_audit logs detailed information about the execution of a stage in the GPU pipeline, including operator details, input/output summaries, and performance metrics.
void log_stage_audit(const op::sirius_physical_operator& op,
                     const op::operator_data& input_data,
                     const op::operator_data& output_data,
                     const sirius_pipeline* pipeline,
                     uint64_t task_id,
                     size_t num_operators,
                     int actual_gpu,
                     std::chrono::microseconds duration)
{
  auto input_summary  = summarize_stage_data(input_data);
  auto output_summary = summarize_stage_data(output_data);
  double byte_ratio   = input_summary.bytes == 0 ? 0.0
                                                 : static_cast<double>(output_summary.bytes) /
                                                   static_cast<double>(input_summary.bytes);
  double row_ratio    = input_summary.rows == 0 ? 0.0
                                                : static_cast<double>(output_summary.rows) /
                                                 static_cast<double>(input_summary.rows);

  SIRIUS_LOG_INFO(
    "[stage-audit] pipeline_id={} task_id={} operator_id={} operator_name={} stage_kind={} "
    "operator_type={} input_type={} output_type={} actual_gpu={} pipeline_operator_count={} "
    "input_batches={} input_rows={} input_columns={} input_bytes={} output_batches={} "
    "output_rows={} output_columns={} output_bytes={} byte_ratio={:.6f} row_ratio={:.6f} "
    "duration_us={}",
    pipeline ? pipeline->get_pipeline_id() : 0,
    task_id,
    op.get_operator_id(),
    op.get_name(),
    stage_kind(op.type),
    static_cast<int>(op.type),
    operator_data_type_name(input_data.get_type()),
    operator_data_type_name(output_data.get_type()),
    actual_gpu,
    num_operators,
    input_summary.batches,
    input_summary.rows,
    input_summary.columns,
    input_summary.bytes,
    output_summary.batches,
    output_summary.rows,
    output_summary.columns,
    output_summary.bytes,
    byte_ratio,
    row_ratio,
    duration.count());
}


// wdy start
// retained_join_batch represents a data batch that is retained in memory for potential reuse in join operations, along with its size in bytes.
struct retained_join_batch {
  std::shared_ptr<cucascade::data_batch> batch;
  std::size_t bytes = 0;
};

std::mutex retained_join_batches_mutex;
std::vector<retained_join_batch> retained_join_batches;
std::size_t retained_join_bytes = 0;

// cached_join_output represents the cached output of a join operation, including the retained data batches, their total size in bytes, and the number of times this cached output has been reused (hits).
struct cached_join_output {
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  std::size_t bytes = 0;
  std::size_t hits  = 0;
};

std::unordered_map<std::string, cached_join_output> retained_join_outputs_by_signature;

// Returns true if the environment variable `name` is set to a truthy value ("1", "true", "on"), false otherwise.
bool env_truthy(const char* name)
{
  auto* value = std::getenv(name);
  if (value == nullptr) { return false; }
  std::string_view sv(value);
  return sv == "1" || sv == "true" || sv == "TRUE" || sv == "on" || sv == "ON";
}

// Returns the value of the environment variable `name` as a size_t, or `fallback` if the variable is not set or cannot be converted to a size_t.
std::size_t env_size_or_default(const char* name, std::size_t fallback)
{
  auto* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') { return fallback; }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return fallback;
  }
}
// Returns true if join output retention is enabled either via configuration or environment variable, false otherwise.
bool join_output_retention_enabled()
{
  return duckdb::Config::JOIN_OUTPUT_RETENTION || env_truthy("SIRIUS_JOIN_OUTPUT_RETENTION");
}
// Returns true if join output reuse is enabled either via configuration or environment variable, false otherwise.
bool join_output_reuse_enabled()
{
  return duckdb::Config::JOIN_OUTPUT_REUSE || env_truthy("SIRIUS_JOIN_OUTPUT_REUSE");
}
// Returns the configured limit for join output retention in bytes, either from the environment variable or the default configuration.
std::size_t join_output_retention_limit_bytes()
{
  return env_size_or_default("SIRIUS_JOIN_OUTPUT_RETENTION_LIMIT_BYTES",
                             duckdb::Config::JOIN_OUTPUT_RETENTION_LIMIT_BYTES);
}
// Returns the configured maximum batch size for join output retention in bytes, either from the environment variable or the default configuration.
std::size_t join_output_retention_max_batch_bytes()
{
  return env_size_or_default("SIRIUS_JOIN_OUTPUT_RETENTION_MAX_BATCH_BYTES",
                             duckdb::Config::JOIN_OUTPUT_RETENTION_MAX_BATCH_BYTES);
}
// Clears all retained join batches and cached outputs, logging the reason for the clearance and the number of batches and bytes released.
void clear_retained_join_batches_locked(const char* reason)
{
  auto released_batches = retained_join_batches.size();
  auto released_bytes   = retained_join_bytes;
  retained_join_batches.clear();
  retained_join_outputs_by_signature.clear();
  retained_join_bytes = 0;
  if (released_batches > 0 || released_bytes > 0) {
    SIRIUS_LOG_INFO("[join-retention] cleared reason={} released_batches={} released_bytes={}",
                    reason,
                    released_batches,
                    released_bytes);
  }
}
// Retains the given join output batches for potential reuse, respecting configured limits on total bytes and maximum batch size. Returns true if the batches were retained, false otherwise.
void append_type_signature(std::ostringstream& out, duckdb::vector<sirius::logical_type> const& types)
{
  out << "types(" << types.size() << ")=";
  for (auto const& type : types) {
    auto cudf_type = sirius::get_cudf_type(type);
    out << static_cast<int>(cudf_type.id()) << ":";
  }
}
// Appends a string representation of the operator's signature, including its type, name, estimated cardinality, output types, and any relevant details for scan or join operators. This is used for caching and reuse of join outputs.
void append_operator_signature(std::ostringstream& out, const op::sirius_physical_operator& op)
{
  out << "op{" << static_cast<int>(op.type) << "," << op.get_name() << ",card="
      << op.estimated_cardinality << ",";
  append_type_signature(out, op.get_types());

  if (auto const* scan_op = dynamic_cast<const op::scan::sirius_gpu_scan_operator*>(&op)) {
    out << ",scan_paths=";
    try {
      auto paths = scan_op->get_ingestible().table_info().file_paths();
      for (auto const& path : paths) { out << path << ";"; }
    } catch (...) {
      out << "unprepared";
    }
  }

  if (auto const* join_op = dynamic_cast<const op::sirius_physical_hash_join*>(&op)) {
    out << ",join_type=" << static_cast<int>(join_op->join_type);
    out << ",conditions=";
    for (auto const& cond : join_op->conditions) {
      out << static_cast<int>(cond.comparison) << ";";
    }
    out << ",lhs_cols=";
    for (auto col : join_op->lhs_output_columns.col_idxs) { out << col << ";"; }
    out << ",rhs_cols=";
    for (auto col : join_op->rhs_output_columns.col_idxs) { out << col << ";"; }
  }

  out << ",children=[";
  for (auto const& child : op.children) {
    if (child) { append_operator_signature(out, *child); }
  }
  out << "]}";
}
// Appends a string representation of the input data's shape and types, including the number of rows, columns, bytes, and column data types for each batch. This is used for caching and reuse of join outputs.
void append_input_shape_signature(std::ostringstream& out, const op::operator_data& data)
{
  out << "input{" << static_cast<int>(data.get_type()) << ":";
  auto const* pipelineable = dynamic_cast<const op::pipelineable_operator_data*>(&data);
  if (pipelineable == nullptr) {
    out << "non_pipelineable}";
    return;
  }
  for (auto const& batch : pipelineable->get_read_only_batches(false)) {
    if (!batch.get_data()) { continue; }
    auto view = get_cudf_table_view(batch);
    out << "b(rows=" << view.num_rows() << ",cols=" << view.num_columns()
        << ",bytes=" << batch.get_data()->get_size_in_bytes() << ",types=";
    for (cudf::size_type i = 0; i < view.num_columns(); i++) {
      out << static_cast<int>(view.column(i).type().id()) << ";";
    }
    out << ")";
  }
  out << "}";
}

std::string make_join_reuse_signature(const op::sirius_physical_operator& op,
                                      const op::operator_data& input_data)
{
  std::ostringstream out;
  append_operator_signature(out, op);
  out << "|";
  append_input_shape_signature(out, input_data);
  return out.str();
}

std::unique_ptr<op::operator_data> try_reuse_join_output_if_enabled(
  const op::sirius_physical_operator& op,
  const op::operator_data& input_data,
  const sirius_pipeline* pipeline,
  uint64_t task_id,
  rmm::cuda_stream_view stream,
  bool& reused)
{
  reused = false;
  if (!join_output_reuse_enabled()) { return nullptr; }
  if (op.type != sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) { return nullptr; }

  auto signature = make_join_reuse_signature(op, input_data);
  std::lock_guard<std::mutex> lock(retained_join_batches_mutex);
  auto it = retained_join_outputs_by_signature.find(signature);
  if (it == retained_join_outputs_by_signature.end()) {
    SIRIUS_LOG_INFO("[join-reuse] miss pipeline_id={} task_id={} operator_id={} operator_name={} key_hash={} cache_entries={}",
                    pipeline ? pipeline->get_pipeline_id() : 0,
                    task_id,
                    op.get_operator_id(),
                    op.get_name(),
                    std::hash<std::string>{}(signature),
                    retained_join_outputs_by_signature.size());
    return nullptr;
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> copied_batches;
  copied_batches.reserve(it->second.batches.size());
  auto const executing_device = current_gpu_id();
  for (auto const& cached_batch : it->second.batches) {
    if (!cached_batch) { continue; }
    auto ro = cached_batch->to_read_only();
    if (!ro.get_data() || ro.get_memory_space() == nullptr) { continue; }
    auto const cached_device = ro.get_memory_space()->get_device_id();
    if (cached_device != executing_device) {
      SIRIUS_LOG_INFO(
        "[join-reuse] miss pipeline_id={} task_id={} operator_id={} operator_name={} key_hash={} "
        "cache_entries={} reason=device_mismatch cached_device={} executing_device={}",
        pipeline ? pipeline->get_pipeline_id() : 0,
        task_id,
        op.get_operator_id(),
        op.get_name(),
        std::hash<std::string>{}(signature),
        retained_join_outputs_by_signature.size(),
        cached_device,
        executing_device);
      return nullptr;
    }
    auto view = get_cudf_table_view(ro);
    auto copied_table = std::make_unique<cudf::table>(
      view, stream, ro.get_memory_space()->get_default_allocator());
    copied_batches.push_back(make_data_batch(std::move(copied_table), *ro.get_memory_space(), stream));
  }

  it->second.hits++;
  reused = true;
  SIRIUS_LOG_INFO("[join-reuse] hit pipeline_id={} task_id={} operator_id={} operator_name={} key_hash={} batches={} bytes={} hits={} mode=copy",
                  pipeline ? pipeline->get_pipeline_id() : 0,
                  task_id,
                  op.get_operator_id(),
                  op.get_name(),
                  std::hash<std::string>{}(signature),
                  copied_batches.size(),
                  it->second.bytes,
                  it->second.hits);
  return std::make_unique<op::pipelineable_operator_data>(std::move(copied_batches));
}

void retain_join_output_if_enabled(const op::sirius_physical_operator& op,
                                   const op::operator_data& input_data,
                                   const op::operator_data& output_data,
                                   const sirius_pipeline* pipeline,
                                   uint64_t task_id)
{
  std::lock_guard<std::mutex> lock(retained_join_batches_mutex);
  if (!join_output_retention_enabled()) {
    if (!retained_join_batches.empty()) { clear_retained_join_batches_locked("disabled"); }
    return;
  }

  if (op.type != sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) { return; }
  auto const* pipelineable_output =
    dynamic_cast<const op::pipelineable_operator_data*>(&output_data);
  if (pipelineable_output == nullptr) { return; }

  auto const signature       = make_join_reuse_signature(op, input_data);
  auto const key_hash        = std::hash<std::string>{}(signature);
  auto const limit_bytes     = join_output_retention_limit_bytes();
  auto const max_batch_bytes = join_output_retention_max_batch_bytes();
  std::vector<std::shared_ptr<cucascade::data_batch>> retained_for_signature;
  std::size_t retained_for_signature_bytes = 0;
  for (const auto& batch : pipelineable_output->get_data_batches()) {
    if (!batch) { continue; }
    auto ro = batch->to_read_only();
    auto* data = ro.get_data();
    if (data == nullptr) { continue; }
    auto const bytes = data->get_size_in_bytes();
    auto* space      = ro.get_memory_space();
    auto const tier  = space ? tier_name(space->get_tier()) : "NONE";
    auto const dev   = space ? space->get_device_id() : -1;
    auto const batch_id = batch->get_batch_id();

    if (max_batch_bytes > 0 && bytes > max_batch_bytes) {
      SIRIUS_LOG_INFO(
        "[join-retention] skipped pipeline_id={} task_id={} operator_id={} operator_name={} "
        "batch_id={} bytes={} tier={} device={} cache_bytes={} cache_batches={} limit_bytes={} "
        "max_batch_bytes={} reason=max_batch",
        pipeline ? pipeline->get_pipeline_id() : 0,
        task_id,
        op.get_operator_id(),
        op.get_name(),
        batch_id,
        bytes,
        tier,
        dev,
        retained_join_bytes,
        retained_join_batches.size(),
        limit_bytes,
        max_batch_bytes);
      continue;
    }

    if (limit_bytes > 0 && retained_join_bytes + bytes > limit_bytes) {
      SIRIUS_LOG_INFO(
        "[join-retention] skipped pipeline_id={} task_id={} operator_id={} operator_name={} "
        "batch_id={} bytes={} tier={} device={} cache_bytes={} cache_batches={} limit_bytes={} "
        "max_batch_bytes={} reason=limit",
        pipeline ? pipeline->get_pipeline_id() : 0,
        task_id,
        op.get_operator_id(),
        op.get_name(),
        batch_id,
        bytes,
        tier,
        dev,
        retained_join_bytes,
        retained_join_batches.size(),
        limit_bytes,
        max_batch_bytes);
      continue;
    }

    // Keep the GPU data object alive for the experiment without changing Sirius task
    // subscriber accounting. subscribe()/unsubscribe() are used for pipeline consumers.
    retained_join_batches.push_back(retained_join_batch{batch, bytes});
    retained_join_bytes += bytes;
    retained_for_signature.push_back(batch);
    retained_for_signature_bytes += bytes;
    SIRIUS_LOG_INFO(
      "[join-retention] retained pipeline_id={} task_id={} operator_id={} operator_name={} "
      "batch_id={} bytes={} tier={} device={} cache_bytes={} cache_batches={} limit_bytes={} "
      "max_batch_bytes={}",
      pipeline ? pipeline->get_pipeline_id() : 0,
      task_id,
      op.get_operator_id(),
      op.get_name(),
      batch_id,
      bytes,
      tier,
      dev,
      retained_join_bytes,
      retained_join_batches.size(),
      limit_bytes,
      max_batch_bytes);
  }

  if (!retained_for_signature.empty()) {
    auto& entry = retained_join_outputs_by_signature[signature];
    if (entry.batches.empty()) {
      entry.batches = retained_for_signature;
      entry.bytes   = retained_for_signature_bytes;
      SIRIUS_LOG_INFO("[join-reuse] cached pipeline_id={} task_id={} operator_id={} operator_name={} key_hash={} batches={} bytes={} cache_entries={}",
                      pipeline ? pipeline->get_pipeline_id() : 0,
                      task_id,
                      op.get_operator_id(),
                      op.get_name(),
                      key_hash,
                      entry.batches.size(),
                      entry.bytes,
                      retained_join_outputs_by_signature.size());
    }
  }
}
// wdy end

void log_operator_data(const op::sirius_physical_operator& op,
                       const op::operator_data& data,
                       const sirius_pipeline* pipeline,
                       uint64_t task_id,
                       const char* label,
                       const std::string& extra_info = "")
{
  std::string batch_rows = "";
  size_t total_bytes     = 0;
  size_t num_batches     = 0;

  if (auto* p_data = dynamic_cast<const op::pipelineable_operator_data*>(&data)) {
    const auto& batches = p_data->get_read_only_batches();
    num_batches         = batches.size();
    for (auto const& batch : batches) {
      if (batch.get_data()) {
        auto view = get_cudf_table_view(batch);
        batch_rows += std::to_string(view.num_rows()) + "  ";
        total_bytes += batch.get_data()->get_size_in_bytes();
      }
    }
  } else {
    SIRIUS_LOG_TRACE(
      "[GPU:{}] Pipeline {}: operator {} (id={}) task={} {} non-pipelineable data. {}",
      current_gpu_id(),
      pipeline->get_pipeline_id(),
      op.get_name(),
      op.get_operator_id(),
      task_id,
      label,
      extra_info);
    return;
  }

  SIRIUS_LOG_TRACE(
    "[GPU:{}] Pipeline {}: operator {} (id={}) task={} {} {} batches, num rows: {}, "
    "size: {} bytes ({:.2f} MB). {}",
    current_gpu_id(),
    pipeline->get_pipeline_id(),
    op.get_name(),
    op.get_operator_id(),
    task_id,
    label,
    num_batches,
    batch_rows,
    total_bytes,
    static_cast<double>(total_bytes) / (1024.0 * 1024.0),
    extra_info);
}

std::unique_ptr<op::operator_data> run_one_operator(
  op::sirius_physical_operator& op,
  const op::operator_data& operator_input_data,
  rmm::cuda_stream_view stream,
  const sirius_pipeline* pipeline,
  uint64_t task_id,
  size_t num_operators,
  cucascade::memory::reservation_aware_resource_adaptor* allocator)
{
  log_operator_data(op, operator_input_data, pipeline, task_id, "executing on");

  auto nvtx_label = std::format(
    "Pipeline {}: {} (id={})", pipeline->get_pipeline_id(), op.get_name(), op.get_operator_id());
  nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};
  auto start = std::chrono::high_resolution_clock::now();
  bool reused_join_output = false;
  auto operator_output_data =
    try_reuse_join_output_if_enabled(op, operator_input_data, pipeline, task_id, stream, reused_join_output);
  if (!operator_output_data) {
    operator_output_data = op.execute(operator_input_data, stream);
    stream.synchronize();
  }
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // wdy start
  log_stage_audit(op,
                  operator_input_data,
                  *operator_output_data,
                  pipeline,
                  task_id,
                  num_operators,
                  current_gpu_id(),
                  duration);
  if (!reused_join_output) {
    retain_join_output_if_enabled(op, operator_input_data, *operator_output_data, pipeline, task_id);
  }
  // wdy end

  auto peak_bytes        = allocator ? allocator->get_peak_allocated_bytes(stream) : 0;
  std::string extra_info = fmt::format(
    "execution time: {:.2f} ms, "
    "peak allocated: {} bytes ({:.2f} MB)",
    duration.count() / 1000.0,
    peak_bytes,
    static_cast<double>(peak_bytes) / (1024.0 * 1024.0));
  log_operator_data(op, *operator_output_data, pipeline, task_id, "produced", extra_info);

  validate_operator_output_types(operator_output_data.get(), op);
  return operator_output_data;
}

}  // namespace

gpu_pipeline_task::gpu_pipeline_task(
  uint64_t task_id,
  std::vector<cucascade::shared_data_repository*> data_repos,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : sirius_pipeline_itask(task_id, std::move(local_state), std::move(global_state)),
    _data_repos(std::move(data_repos))
{
  // Subscribe to all input data_batches
  auto& ls = _local_state->cast<gpu_pipeline_task_local_state>();
  if (ls._input_data) {
    auto* pipelineable_input =
      dynamic_cast<const op::pipelineable_operator_data*>(ls._input_data.get());
    if (pipelineable_input) {
      for (const auto& batch : pipelineable_input->get_data_batches()) {
        if (batch) {
          batch->subscribe();
          _input_batches.push_back(batch);
        }
      }
    }
  }
  if (auto* pipeline = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline()) {
    pipeline->mark_task_created();
  }
}

gpu_pipeline_task::~gpu_pipeline_task()
{
  // Unsubscribe from all input data_batches
  for (const auto& batch : _input_batches) {
    if (batch) {
      try {
        batch->unsubscribe();
      } catch (...) {
        // Destructor must not throw; log if possible
        SIRIUS_LOG_WARN("gpu_pipeline_task: unsubscribe failed for batch {}",
                        batch->get_batch_id());
      }
    }
  }

  if (_global_state == nullptr ||
      _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline() == nullptr) {
    return;
  }
  _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline()->mark_task_completed();
}

const sirius_pipeline* gpu_pipeline_task::get_pipeline() const
{
  return _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
}

std::unique_ptr<op::operator_data> gpu_pipeline_task::compute_task(rmm::cuda_stream_view stream)
{
  auto pipeline     = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();
  auto operator_input_output_data = std::move(local_state._input_data);
  auto operators                  = pipeline->get_operators();
  auto start_index                = local_state._start_operator_index;

  if (start_index > 0) {
    SIRIUS_LOG_INFO("Pipeline {}: resuming task {} from operator index {} (of {})",
                    pipeline->get_pipeline_id(),
                    get_task_id(),
                    start_index,
                    operators.size());
  }

  auto executor_thread_resource_id = uuid::new_nil();
  if (telemetry::executor_thread_telemetry_handle.has_value()) {
    executor_thread_resource_id = telemetry::executor_thread_telemetry_handle->handle->uuid();
  } else {
    SIRIUS_LOG_ERROR(
      "gpu_pipeline_task::execute_operator: executor thread telemetry handle is not "
      "initialized");
  }

  for (size_t i = start_index; i < operators.size(); i++) {
    auto& op = operators[i].get();
    try {
      this->telemetry_handle().computing({
        .instance_name       = "",
        .current_operator_id = static_cast<uint32_t>(
          op.get_operator_id()),  // TODO(dhruv9vats): look into possible overflow
        .input_bytes                 = operator_input_output_data->get_estimated_size_in_bytes(),
        .executor_thread_resource_id = executor_thread_resource_id,
      });
      operator_input_output_data = run_one_operator(
        op, *operator_input_output_data, stream, pipeline, _task_id, operators.size(), _allocator);
    } catch (const rmm::out_of_memory& oom) {
      auto peak_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0;
      // Subtract the peak allocated bytes to the input data to get the peak allocated bytes for the
      // operators, clamping to zero to avoid unsigned underflow.
      auto const bytes_to_materialize_input =
        local_state.get_reservation_size_info()->bytes_to_materialize_input;
      if (peak_bytes > bytes_to_materialize_input) {
        peak_bytes -= bytes_to_materialize_input;
      } else {
        peak_bytes = 0;
      }
      size_t requested_bytes = 0;
      size_t global_usage    = 0;
      if (auto const* cc_oom =
            dynamic_cast<const cucascade::memory::cucascade_out_of_memory*>(&oom)) {
        requested_bytes = cc_oom->requested_bytes;
        global_usage    = cc_oom->global_usage;
      }
      size_t reservation_bytes =
        _local_state->cast<gpu_pipeline_task_local_state>().get_reservation_bytes();
      SIRIUS_LOG_WARN(
        "Pipeline {}: OOM at operator {} (id={}, index {}/{}), "
        "requested {} bytes ({:.2f} MB), global usage {} bytes ({:.2f} MB), "
        "peak allocated {} bytes ({:.2f} MB), "
        "bytes to materialize input {} bytes ({:.2f} MB), "
        "reservation {} bytes ({:.2f} MB), "
        "rescheduling task {}",
        pipeline->get_pipeline_id(),
        op.get_name(),
        op.get_operator_id(),
        i,
        operators.size(),
        requested_bytes,
        static_cast<double>(requested_bytes) / (1024.0 * 1024.0),
        global_usage,
        static_cast<double>(global_usage) / (1024.0 * 1024.0),
        peak_bytes,
        static_cast<double>(peak_bytes) / (1024.0 * 1024.0),
        local_state.get_reservation_size_info()->bytes_to_materialize_input,
        static_cast<double>(local_state.get_reservation_size_info()->bytes_to_materialize_input) /
          (1024.0 * 1024.0),
        reservation_bytes,
        static_cast<double>(reservation_bytes) / (1024.0 * 1024.0),
        get_task_id());

      auto input_basis = _local_state->cast<gpu_pipeline_task_local_state>()
                           .get_reservation_size_info()
                           ->input_basis;
      auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
      global.get_memory_history().record_on_failure(input_basis, peak_bytes);

      throw oom_reschedule_exception(
        std::move(operator_input_output_data),
        i,
        "OOM at operator " + op.get_name() + " (index " + std::to_string(i) + ")");
    }
  }

  return operator_input_output_data;
}

void gpu_pipeline_task::publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream)
{
  auto pipeline       = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
  auto sink_operators = pipeline->get_sink();
  if (sink_operators) {
    auto nvtx_label = std::format("Pipeline {}: {} (id={}) sink",
                                  pipeline->get_pipeline_id(),
                                  sink_operators->get_name(),
                                  sink_operators->get_operator_id());
    nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};
    auto const sink_start = std::chrono::high_resolution_clock::now();
    sink_operators.get()->sink(output_data, stream);
    auto const sink_end = std::chrono::high_resolution_clock::now();
    auto const sink_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(sink_end - sink_start);
    SIRIUS_LOG_TRACE("Pipeline {}: operator {} (id={}) sink execution time: {:.2f} ms",
                     pipeline->get_pipeline_id(),
                     sink_operators->get_name(),
                     sink_operators->get_operator_id(),
                     sink_duration.count() / 1000.0);
  } else {
    throw std::runtime_error("Sink operator not found");
  }
}

void gpu_pipeline_task::execute(rmm::cuda_stream_view stream)
{
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();
  auto pipeline     = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
  auto operators    = pipeline->get_operators();
  auto& first_op    = operators[local_state._start_operator_index].get();

  std::string op_chain;
  auto source_op = pipeline->get_source();
  if (source_op) { op_chain += std::format("{} -> ", source_op->get_name()); }
  for (size_t i = 0; i < operators.size(); i++) {
    op_chain += operators[i].get().get_name();
    if (i + 1 < operators.size()) { op_chain += " -> "; }
  }
  auto sink_op = pipeline->get_sink();
  if (sink_op) { op_chain += std::format(" -> {}", sink_op->get_name()); }
  auto nvtx_label =
    std::format("Pipeline {} Task {} [{}]", pipeline->get_pipeline_id(), get_task_id(), op_chain);
  nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};

  auto const prepare_start = std::chrono::high_resolution_clock::now();
  auto reservation         = local_state.release_reservation();
  if (!reservation) { throw std::runtime_error("GPU pipeline task requires a memory reservation"); }
  auto reservation_bytes = reservation->size();
  const auto* requested_memory_space =
    reservation != nullptr ? &reservation->get_memory_space() : nullptr;
  auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  allocator->attach_reservation_to_tracker(
    stream, std::move(reservation), nullptr, std::make_unique<memory::defragmenter_oom_policy>());
  absl::Cleanup source_closer = [allocator, stream]() {
    allocator->reset_stream_reservation(stream);
  };

  if (!local_state._input_data) {
    throw std::runtime_error("gpu_pipeline_task::execute: input_data is null");
  }

  auto executor_thread_resource_id = uuid::new_nil();
  if (telemetry::executor_thread_telemetry_handle.has_value()) {
    executor_thread_resource_id = telemetry::executor_thread_telemetry_handle->handle->uuid();
  } else {
    SIRIUS_LOG_ERROR(
      "gpu_pipeline_task::execute: executor thread telemetry handle is not initialized");
  }
  telemetry_handle().preparing({
    .instance_name               = "",
    .target_tier                 = "GPU",
    .executor_thread_resource_id = executor_thread_resource_id,
  });
  // wdy start
  if (auto* pipelineable_input =
        dynamic_cast<const op::pipelineable_operator_data*>(local_state._input_data.get())) {
    log_locality_snapshot("prepare_before",
                          pipeline,
                          get_task_id(),
                          get_preferred_device_id(),
                          current_gpu_id(),
                          requested_memory_space,
                          summarize_unlocked_batches(*pipelineable_input, requested_memory_space));
  }
  // wdy end

  try {
    local_state._input_data->prepare_for_processing(requested_memory_space, stream);
    // synchronizing here to ensure the timing collected by Quent and logging for preparing the task
    // is accurate.
    stream.synchronize();
    // wdy start
    if (auto* pipelineable_input =
          dynamic_cast<const op::pipelineable_operator_data*>(local_state._input_data.get())) {
      log_locality_snapshot("prepare_after",
                            pipeline,
                            get_task_id(),
                            get_preferred_device_id(),
                            current_gpu_id(),
                            requested_memory_space,
                            summarize_locked_batches(*pipelineable_input, requested_memory_space));
    }
    // wdy end
  } catch (const rmm::out_of_memory& oom) {
    auto peak_bytes  = allocator->get_peak_allocated_bytes(stream);
    auto input_basis = local_state.get_reservation_size_info()->input_basis;
    auto& global     = _global_state->cast<gpu_pipeline_task_global_state>();
    global.get_memory_history().record_on_failure(input_basis, peak_bytes);

    SIRIUS_LOG_ERROR("Pipeline {}: OOM preparing batches for processing",
                     pipeline->get_pipeline_id());
    throw oom_reschedule_exception(
      std::move(local_state._input_data),
      0,
      std::string("OOM while preparing batches for processing: ") + oom.what());
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("Unknown error in prepare_for_processing for pipeline {}: {}",
                     pipeline->get_pipeline_id(),
                     e.what());
    throw;
  }

  auto const prepare_end = std::chrono::high_resolution_clock::now();
  auto const prepare_duration =
    std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - prepare_start);
  SIRIUS_LOG_TRACE("Pipeline {}: operator {} (id={}) prepare execution time: {:.2f} ms",
                   pipeline->get_pipeline_id(),
                   first_op.get_name(),
                   first_op.get_operator_id(),
                   prepare_duration.count() / 1000.0);

  // All input batches are now locked for reading via _read_only_data_batches inside
  // local_state._input_data. The locks are released when the pipelineable_operator_data
  // is destroyed after the first operator's execute() consumes it.

  // 2. Set reservation_aware_memory_resource_ref as the default cudf allocator
  // 3. Execute cudf operators on the pipeline
  _allocator       = allocator;
  auto input_basis = local_state.get_reservation_size_info()->input_basis;
  std::unique_ptr<op::operator_data> output_data = compute_task(stream);

  // Record memory metrics for future reservation estimates
  if (output_data) {
    auto peak_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0;
    // Subtract the peak allocated bytes to the input data to get the peak allocated bytes for the
    // operators. Clamp at zero to avoid size_t underflow when estimates exceed the observed peak.
    if (peak_bytes > local_state.get_reservation_size_info()->bytes_to_materialize_input) {
      peak_bytes -= local_state.get_reservation_size_info()->bytes_to_materialize_input;
    } else {
      peak_bytes = 0;
    }
    std::size_t output_bytes = 0;
    auto* pipelineable_output =
      dynamic_cast<const op::pipelineable_operator_data*>(output_data.get());
    if (pipelineable_output) {
      for (const auto& batch : pipelineable_output->get_read_only_batches(false)) {
        output_bytes += batch.get_data()->get_size_in_bytes();
      }
    }
    auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
    global.get_memory_history().record({input_basis, peak_bytes, output_bytes});
    SIRIUS_LOG_TRACE(
      "[GPU:{}] Pipeline {}: memory history record - task={}, input_basis={}, output_bytes={}, "
      "reservation_bytes={}, peak_bytes={}, peak_bytes_to_materialize_input={}",
      current_gpu_id(),
      pipeline->get_pipeline_id(),
      _task_id,
      input_basis,
      output_bytes,
      reservation_bytes,
      peak_bytes,
      local_state.get_reservation_size_info()->bytes_to_materialize_input);
  }

  if (output_data) { publish_output(*output_data, stream); }

  // The input pipelineable_operator_data (with its _read_only_data_batches) was destroyed
  // when compute_task replaced operator_input_output_data, releasing all shared locks.
}

std::size_t gpu_pipeline_task::get_input_size() const
{
  auto& local_state      = _local_state->cast<gpu_pipeline_task_local_state>();
  std::size_t input_size = 0;
  if (!local_state._input_data) { return 0; }
  auto* pipelineable_input =
    dynamic_cast<const op::pipelineable_operator_data*>(local_state._input_data.get());
  if (!pipelineable_input) { return 0; }
  for (const auto& batch : pipelineable_input->get_read_only_batches(false)) {
    input_size += batch.get_data()->get_size_in_bytes();
  }
  return input_size;
}

pipeline::reservation_size_info gpu_pipeline_task::get_estimated_reservation_size_info() const
{
  auto& ls                         = _local_state->cast<gpu_pipeline_task_local_state>();
  auto& gs                         = _global_state->cast<gpu_pipeline_task_global_state>();
  std::size_t input_basis          = ls.get_task_consumption_basis();
  std::size_t bytes_to_materialize = ls.get_estimated_bytes_to_materialize_input();
  auto peak_opt                    = gs.get_memory_history().estimate_peak_memory(input_basis);

  pipeline::reservation_size_info info;
  info.input_basis                = input_basis;
  info.bytes_to_materialize_input = bytes_to_materialize;
  info.had_history                = peak_opt.has_value();

  if (peak_opt.has_value()) {
    info.peak_memory_estimate = *peak_opt;
  } else {
    std::size_t num_batches = 0;
    if (auto* pd = dynamic_cast<const op::pipelineable_operator_data*>(ls._input_data.get())) {
      num_batches = pd->get_data_batches().size();
    }
    const auto input_type =
      ls._input_data ? ls._input_data->get_type() : op::operator_data_type::BASE;
    const bool input_resident = ls._input_data && ls._input_data->is_resident();
    const op::input_stats stats{num_batches, input_basis, input_type, input_resident};

    std::size_t max_estimate = 0;
    if (auto* pipeline = gs.get_pipeline()) {
      for (auto& op_ref : pipeline->get_operators()) {
        max_estimate = std::max(max_estimate, op_ref.get().no_history_peak_memory_estimate(stats));
      }
    }
    // If every operator returned 0 (all pass-throughs), fall back to the 2× default.
    info.peak_memory_estimate = (max_estimate > 0) ? max_estimate : (input_basis * 2);
  }

  info.reservation_size = info.peak_memory_estimate + bytes_to_materialize;
  return info;
}

std::vector<op::sirius_physical_operator*> gpu_pipeline_task::get_output_consumers()
{
  std::vector<op::sirius_physical_operator*> output_consumers;
  if (_global_state == nullptr ||
      _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline() == nullptr) {
    return output_consumers;
  }
  return _global_state->cast<gpu_pipeline_task_global_state>()
    .get_pipeline()
    ->get_output_consumers();
}

std::unique_ptr<gpu_pipeline_task> gpu_pipeline_task::create_rescheduled_task(
  uint64_t task_id, std::unique_ptr<sirius_pipeline_task_local_state> local_state)
{
  return std::make_unique<gpu_pipeline_task>(
    task_id, _data_repos, std::move(local_state), get_shared_global_state());
}

}  // namespace pipeline
}  // namespace sirius
