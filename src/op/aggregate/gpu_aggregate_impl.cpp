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

#include "op/aggregate/gpu_aggregate_impl.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <chrono>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <unordered_set>

namespace sirius {
namespace op {

namespace {

// Any subset (a group) is bounded by row_count times the column extrema. Check
// with 128-bit host arithmetic so the proof itself cannot overflow. Nulls do
// not contribute; using all rows rather than valid rows is conservative.
bool average_sum_fits_native(cudf::column_view const& column,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto extrema = cudf::minmax(column, stream, mr);
  if (!extrema.first->is_valid(stream)) { return true; }
  auto const rows = static_cast<__int128>(column.size());
  if (column.type().id() == cudf::type_id::INT64) {
    auto const low =
      static_cast<cudf::numeric_scalar<int64_t> const&>(*extrema.first).value(stream);
    auto const high =
      static_cast<cudf::numeric_scalar<int64_t> const&>(*extrema.second).value(stream);
    return rows * low >= std::numeric_limits<int64_t>::min() &&
           rows * high <= std::numeric_limits<int64_t>::max();
  }
  auto const high =
    static_cast<cudf::numeric_scalar<uint64_t> const&>(*extrema.second).value(stream);
  return rows * static_cast<__int128>(high) <=
         static_cast<__int128>(std::numeric_limits<uint64_t>::max());
}

}  // namespace

template <typename Base = cudf::aggregation>
std::unique_ptr<Base> get_local_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::MIN: return cudf::make_min_aggregation<Base>();
    case cudf::aggregation::Kind::MAX: return cudf::make_max_aggregation<Base>();
    case cudf::aggregation::Kind::COUNT_ALL:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::Kind::COUNT_VALID:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::Kind::SUM: return cudf::make_sum_aggregation<Base>();
    default:
      throw std::runtime_error("Unsupported cudf aggregate kind in `get_local_aggregation()`: " +
                               std::to_string(static_cast<int>(kind)));
  }
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_ungrouped_aggregate(
  const cucascade::read_only_data_batch& input,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_ungrouped_aggregate()`");
  }
  std::vector<std::unique_ptr<cudf::column>> output_cols;
  auto input_table = get_cudf_table_view(input);
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& input_col       = input_table.column(aggregate_idx[i]);
    auto reduce_aggregation     = get_local_aggregation<cudf::reduce_aggregation>(aggregates[i]);
    cudf::data_type output_type = input_col.type();
    switch (aggregates[i]) {
      case cudf::aggregation::Kind::SUM: {
        switch (output_type.id()) {
          case cudf::type_id::INT8:
          case cudf::type_id::INT16:
          case cudf::type_id::INT32: {
            output_type = cudf::data_type(cudf::type_id::INT64);
            break;
          }
          case cudf::type_id::UINT8:
          case cudf::type_id::UINT16:
          case cudf::type_id::UINT32: {
            output_type = cudf::data_type(cudf::type_id::UINT64);
            break;
          }
          case cudf::type_id::DECIMAL64:
            if (input_col.type().id() == cudf::type_id::DECIMAL64) {
              output_type = cudf::data_type(cudf::type_id::DECIMAL128, output_type.scale());
            }
            break;
          case cudf::type_id::DECIMAL32:
            if (input_col.type().id() == cudf::type_id::DECIMAL32) {
              output_type = cudf::data_type(cudf::type_id::DECIMAL64, output_type.scale());
            }
            break;
          default: break;
        }
        break;
      }
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        output_type = cudf::data_type(cudf::type_id::INT64);
        break;
      }
      default: break;
    }
    auto output_scalar = cudf::reduce(
      input_col, *reduce_aggregation, output_type, stream, memory_space.get_default_allocator());
    output_cols.push_back(cudf::make_column_from_scalar(
      *output_scalar, 1, stream, memory_space.get_default_allocator()));
  }
  auto output_table = std::make_unique<cudf::table>(
    std::move(output_cols), stream, memory_space.get_default_allocator());

  return make_data_batch(std::move(output_table), memory_space, stream);
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_grouped_aggregate(
  const cucascade::read_only_data_batch& input,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  const std::vector<std::vector<int>>& aggregate_struct_col_indices,
  const std::vector<bool>& aggregate_widen_sum,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_grouped_aggregate()`");
  }

  const bool has_struct_col_indices = !aggregate_struct_col_indices.empty();
  const bool has_widen_sum          = !aggregate_widen_sum.empty();

  auto input_table = get_cudf_table_view(input);
  auto mr          = memory_space.get_default_allocator();

  // Synchronize phase boundaries only in dedicated diagnostics.
  static bool const diagnose = [] {
    auto const* value = std::getenv("SIRIUS_GROUPBY_DIAGNOSTIC");
    return value != nullptr && std::string_view(value) == "1";
  }();
  static bool const use_native_average = [] {
    auto const* value = std::getenv("SIRIUS_GROUPBY_SAFE_AVG_INT64");
    return value == nullptr || std::string_view(value) != "0";
  }();
  auto phase_start = std::chrono::steady_clock::now();
  if (diagnose) {
    stream.synchronize();
    phase_start = std::chrono::steady_clock::now();
  }
  auto report_phase = [&](char const* name) {
    if (!diagnose) { return; }
    stream.synchronize();
    auto const now = std::chrono::steady_clock::now();
    SIRIUS_LOG_INFO(
      "[groupby-phase] rows={} phase={} elapsed_us={}",
      input_table.num_rows(),
      name,
      std::chrono::duration_cast<std::chrono::microseconds>(now - phase_start).count());
    phase_start = now;
  };

  // Dictionary-encode STRING group keys when:
  //  1. Average string length >= 4 bytes (short strings hash nearly as fast as
  //     int32, so the encode/decode overhead is not worthwhile), AND
  //  2. NDV / row_count < 10% (high-cardinality columns produce huge
  //     dictionaries that negate the hashing benefit).
  // The avg_len check is O(1) (single offset read) and gates the more
  // expensive HLL-based NDV estimate.
  constexpr double dict_encode_min_avg_len = 8.0;
  constexpr double dict_encode_max_ratio   = 0.10;
  std::vector<std::unique_ptr<cudf::column>> encoded_key_owners;
  std::vector<cudf::column_view> group_cols;
  group_cols.reserve(group_idx.size());
  for (int idx : group_idx) {
    auto col = input_table.column(idx);
    if (col.type().id() == cudf::type_id::STRING && col.size() > 0) {
      cudf::strings_column_view scv(col);
      auto avg_len = static_cast<double>(scv.chars_size(stream)) / col.size();
      if (avg_len >= dict_encode_min_avg_len) {
        cudf::approx_distinct_count adc(cudf::table_view({col}),
                                        12,
                                        cudf::null_policy::EXCLUDE,
                                        cudf::nan_policy::NAN_IS_VALID,
                                        stream);
        auto ndv     = adc.estimate(stream);
        double ratio = static_cast<double>(ndv) / col.size();
        if (ratio < dict_encode_max_ratio) {
          auto encoded =
            cudf::dictionary::encode(col, cudf::data_type{cudf::type_id::INT32}, stream, mr);
          group_cols.push_back(encoded->view());
          encoded_key_owners.push_back(std::move(encoded));
          SIRIUS_LOG_DEBUG(
            "local_grouped_agg: dict-encoding key col {} (avg_len={:.1f}, ndv={}, rows={})",
            idx,
            avg_len,
            ndv,
            col.size());
        } else {
          group_cols.push_back(col);
          SIRIUS_LOG_DEBUG(
            "local_grouped_agg: skipping dict-encode for key col {} "
            "(avg_len={:.1f}, ndv={}, rows={}, ratio={:.4f})",
            idx,
            avg_len,
            ndv,
            col.size(),
            ratio);
        }
      } else {
        group_cols.push_back(col);
        SIRIUS_LOG_DEBUG(
          "local_grouped_agg: skipping dict-encode for key col {} (avg_len={:.1f} < 4.0)",
          idx,
          avg_len);
      }
    } else {
      group_cols.push_back(col);
    }
  }
  report_phase("key_prep");

  // Make aggregation requests, group aggregations on the same column in the single request.
  // For multi-column COLLECT_SET, a synthetic negative key -(i+1) is used so that each such
  // aggregate gets its own request with a freshly synthesized struct column.
  std::unordered_map<int, int> widened_source;  // synthetic key -> source column index
  std::unordered_map<int, std::vector<std::unique_ptr<cudf::groupby_aggregation>>> input_col_to_agg;
  std::unordered_map<int, std::vector<size_t>> input_col_to_output_idx;
  std::vector<int> input_col_order;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& aggregate_kind = aggregates[i];
    int aggregate_col_id;
    if (has_struct_col_indices && !aggregate_struct_col_indices[i].empty()) {
      // Multi-column COLLECT_SET: use a unique synthetic negative key for this slot.
      aggregate_col_id = -(static_cast<int>(i) + 1);
    } else if (has_widen_sum && aggregate_widen_sum[i] &&
               (input_table.column(aggregate_idx[i]).type().id() == cudf::type_id::INT64 ||
                input_table.column(aggregate_idx[i]).type().id() == cudf::type_id::UINT64)) {
      // This SUM is the accumulator behind an AVG over a 64-bit integer column, which has
      // nowhere left to widen into: AVG(UserID) over ClickBench's hits sums to 2.5e26 against
      // an int64 ceiling of 9.2e18. Give it a separate request, using DECIMAL128 unless
      // column extrema prove every group sum fits in the original type. Its partial
      // result is always DECIMAL128, distinct from a plain SUM on the same column,
      // whose INT64 output the merge still expects.
      aggregate_col_id                 = -(static_cast<int>(i) + 1);
      widened_source[aggregate_col_id] = aggregate_idx[i];
    } else {
      aggregate_col_id = aggregate_idx[i];
    }
    if (!input_col_to_agg.contains(aggregate_col_id)) {
      input_col_order.push_back(aggregate_col_id);
    }
    std::unique_ptr<cudf::groupby_aggregation> groupby_aggregation;
    if (aggregate_kind == cudf::aggregation::Kind::COLLECT_SET) {
      groupby_aggregation =
        cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
    } else {
      groupby_aggregation = get_local_aggregation<cudf::groupby_aggregation>(aggregate_kind);
    }
    input_col_to_agg[aggregate_col_id].push_back(std::move(groupby_aggregation));
    input_col_to_output_idx[aggregate_col_id].push_back(i);
  }

  // Temp columns backing synthetic requests (multi-col COLLECT_SET structs and widened
  // AVG accumulators); these must outlive the groupby call.
  std::vector<std::unique_ptr<cudf::column>> temp_struct_cols;

  std::vector<cudf::groupby::aggregation_request> requests;
  std::unordered_set<int> native_average_requests;
  for (int aggregate_col_id : input_col_order) {
    cudf::groupby::aggregation_request request;
    if (auto source_entry = widened_source.find(aggregate_col_id);
        source_entry != widened_source.end()) {
      auto const source = input_table.column(source_entry->second);
      if (use_native_average && average_sum_fits_native(source, stream, mr)) {
        request.values = source;
        native_average_requests.insert(aggregate_col_id);
      } else {
        auto widened   = cudf::cast(source,
                                  cudf::data_type(cudf::type_id::DECIMAL128, 0),
                                  stream,
                                  memory_space.get_default_allocator());
        request.values = widened->view();
        temp_struct_cols.push_back(std::move(widened));
      }
    } else if (aggregate_col_id < 0) {
      // Multi-col COLLECT_SET: synthesize a struct column from the component columns.
      // The synthetic key is -(slot_index + 1), so slot_index = -aggregate_col_id - 1.
      size_t slot_idx            = static_cast<size_t>(-aggregate_col_id - 1);
      const auto& struct_indices = aggregate_struct_col_indices[slot_idx];
      std::vector<std::unique_ptr<cudf::column>> struct_children;
      for (int col_idx : struct_indices) {
        struct_children.push_back(std::make_unique<cudf::column>(
          input_table.column(col_idx), stream, memory_space.get_default_allocator()));
      }
      auto struct_col = cudf::make_structs_column(input_table.num_rows(),
                                                  std::move(struct_children),
                                                  0,
                                                  rmm::device_buffer{},
                                                  stream,
                                                  memory_space.get_default_allocator());
      request.values  = struct_col->view();
      temp_struct_cols.push_back(std::move(struct_col));
    } else {
      request.values = input_table.column(aggregate_col_id);
    }
    request.aggregations = std::move(input_col_to_agg[aggregate_col_id]);
    requests.push_back(std::move(request));
  }

  report_phase("request_prep");
  if (diagnose) {
    for (auto const& request : requests) {
      for (auto const& agg : request.aggregations) {
        SIRIUS_LOG_INFO("[groupby-request] rows={} value_type={} aggregation={}",
                        input_table.num_rows(),
                        static_cast<int>(request.values.type().id()),
                        static_cast<int>(agg->kind));
      }
    }
  }

  cudf::groupby::groupby grpby_obj(cudf::table_view(group_cols), cudf::null_policy::INCLUDE);
  // Call cudf groupby and populate output columns
  auto groupby_result = grpby_obj.aggregate(requests, stream, mr);
  report_phase("aggregate");
  auto output_cols = groupby_result.first->release();

  // Decode dictionary-encoded group key columns back to STRING
  for (size_t i = 0; i < group_idx.size(); i++) {
    if (output_cols[i]->type().id() == cudf::type_id::DICTIONARY32) {
      cudf::dictionary_column_view dict_view(output_cols[i]->view());
      output_cols[i] = cudf::dictionary::decode(dict_view, stream, mr);
    }
  }

  output_cols.resize(group_idx.size() + aggregate_idx.size());
  for (size_t i = 0; i < input_col_order.size(); ++i) {
    int aggregate_col_id     = input_col_order[i];
    auto& aggregation_result = groupby_result.second[i];

    // need to cast count aggregation result to int64 (not applicable for COLLECT_SET)
    if (requests[i].aggregations.size() == 1 &&
        requests[i].aggregations[0]->kind != cudf::aggregation::Kind::COLLECT_SET &&
        (requests[i].aggregations[0]->kind == cudf::aggregation::Kind::COUNT_VALID ||
         requests[i].aggregations[0]->kind == cudf::aggregation::Kind::COUNT_ALL)) {
      if (aggregation_result.results.size() != 1) {
        throw std::runtime_error("Expected 1 result for count aggregation, got " +
                                 std::to_string(aggregation_result.results.size()));
      }
      auto result_view = aggregation_result.results[0]->view();
      if (result_view.type().id() != cudf::type_id::INT64) {
        aggregation_result.results[0] = cudf::cast(result_view,
                                                   cudf::data_type(cudf::type_id::INT64),
                                                   stream,
                                                   memory_space.get_default_allocator());
      }
    }

    const auto& output_idx = input_col_to_output_idx[aggregate_col_id];
    for (size_t j = 0; j < output_idx.size(); ++j) {
      auto result_view = aggregation_result.results[j]->view();
      // Widen decimal result for SUM (expected by duckdb)
      if (requests[i].aggregations[j]->kind == cudf::aggregation::Kind::SUM) {
        if (native_average_requests.contains(aggregate_col_id)) {
          // Keep the local partial-sum schema identical to the overflow-safe
          // path. Different batches may need different accumulator widths;
          // every AVG partial still reaches the merge as DECIMAL128.
          aggregation_result.results[j] = cudf::cast(result_view,
                                                     cudf::data_type(cudf::type_id::DECIMAL128, 0),
                                                     stream,
                                                     memory_space.get_default_allocator());
        } else if (requests[i].values.type().id() == cudf::type_id::DECIMAL64) {
          aggregation_result.results[j] =
            cudf::cast(result_view,
                       cudf::data_type(cudf::type_id::DECIMAL128, result_view.type().scale()),
                       stream,
                       memory_space.get_default_allocator());
        } else if (requests[i].values.type().id() == cudf::type_id::DECIMAL32) {
          aggregation_result.results[j] =
            cudf::cast(result_view,
                       cudf::data_type(cudf::type_id::DECIMAL64, result_view.type().scale()),
                       stream,
                       memory_space.get_default_allocator());
        }
      }
      size_t output_col_id       = group_idx.size() + output_idx[j];
      output_cols[output_col_id] = std::move(aggregation_result.results[j]);
    }
  }

  // Create the output data batch
  auto output_table = std::make_unique<cudf::table>(
    std::move(output_cols), stream, memory_space.get_default_allocator());
  return make_data_batch(std::move(output_table), memory_space, stream);
}

}  // namespace op
}  // namespace sirius
