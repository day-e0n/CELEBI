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
#include "io/cache/types.hpp"

#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <log/logging.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
sirius_gpu_scan_operator::sirius_gpu_scan_operator(duckdb::vector<sirius::logical_type> types,
                                                   duckdb::idx_t estimated_cardinality,
                                                   std::shared_ptr<gpu_ingestible> ingestible)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_SCAN, std::move(types), estimated_cardinality),
    _ingestible(std::move(ingestible)),
    _split_connector(std::make_shared<scan_manager::split_connector>())
{
}

sirius_gpu_scan_operator::~sirius_gpu_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_gpu_scan_operator::all_ports_empty() { return _split_connector->is_closed(); }

std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  if (auto* scan_input = dynamic_cast<scan_operator_input*>(next->get()); scan_input) {
    scan_input->prefetch(io::cache::prefetching_stage::immediate);
  }
  return std::move(*next);
}

//===----------------------------------------------------------------------===//
// scan_manager wiring
//===----------------------------------------------------------------------===//
const ingestible_table_info& sirius_gpu_scan_operator::peek_table_info() const
{
  return _ingestible->table_info();
}

gpu_ingestible& sirius_gpu_scan_operator::get_ingestible() const { return *_ingestible; }

scan_manager::split_connector& sirius_gpu_scan_operator::get_split_connector()
{
  return *_split_connector;
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::execute(
  const op::operator_data& input_data, rmm::cuda_stream_view stream)
{
  auto scan_input = dynamic_cast<const scan_operator_input*>(&input_data);
  if (!scan_input) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator::execute] expected input of type scan_operator_input; got " +
      std::string(typeid(input_data).name()));
  }

  ::cucascade::memory::memory_space* mem_space = scan_input->gpu_memory_space;
  std::unique_ptr<cudf::table> output_table;
  // [scan-audit] covers materialize_table only; time the two stages after it as well.
  auto const t0           = std::chrono::steady_clock::now();
  auto materialized_table = _ingestible->materialize_table(*scan_input, stream);
  auto const t1           = std::chrono::steady_clock::now();
  if (materialized_table.state != filter_state::ROW_FILTERED_AND_PROJECTED) {
    output_table =
      _ingestible->post_filter_and_project(std::move(materialized_table), *mem_space, stream);
  } else {
    output_table = materialized_table.table.release(stream, mem_space->get_default_allocator());
  }
  auto const t2          = std::chrono::steady_clock::now();
  auto const output_rows = output_table ? output_table->num_rows() : 0;

  auto batch    = sirius::make_data_batch(std::move(output_table), *mem_space, stream);
  auto const t3 = std::chrono::steady_clock::now();
  auto const us = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };
  SIRIUS_LOG_INFO("[scan-stage] materialize_us={} postfilter_us={} make_batch_us={} rows={}",
                  us(t0, t1),
                  us(t1, t2),
                  us(t2, t3),
                  output_rows);
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  // Pinned (cached) inputs are pass-throughs in the common case, so the estimate equals the
  // input size. For a fresh read, stats.bytes is already the split's UNCOMPRESSED size (the
  // footer's per-column-chunk sizes), so the factor only has to cover decode temporaries and
  // the post-filter copy -- not decompression. Peak use without history is about 2-3x
  // stats.bytes (SIRIUS_RESERVATION_TRACE=1); over-reserving blocks other tasks in Reserving.
  // The per-pipeline history takes over after the first task.
  // SIRIUS_SCAN_NO_HISTORY_FACTOR overrides the factor.
  static double const factor = [] {
    auto const* v = std::getenv("SIRIUS_SCAN_NO_HISTORY_FACTOR");
    if (v == nullptr || v[0] == '\0') { return 3.0; }
    try {
      auto const parsed = std::stod(v);
      return parsed > 0.0 ? parsed : 3.0;
    } catch (...) {
      return 3.0;
    }
  }();
  if (stats.resident) { return stats.bytes; }
  return static_cast<std::size_t>(static_cast<double>(stats.bytes) * factor);
}

}  // namespace sirius::op::scan
