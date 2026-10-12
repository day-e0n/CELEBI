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

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>

#include <algorithm>
#include <limits>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;

}  // namespace

TEMPLATE_TEST_CASE("grouped AVG keeps DECIMAL128 partial sums for safe and overflowing inputs",
                   "[physical_grouped_aggregate][avg_accumulator]",
                   int64_t,
                   uint64_t)
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto stream  = sirius::test::operator_utils::default_stream();
  auto mr      = sirius::test::operator_utils::get_resource_ref(*space);
  using T      = TestType;
  auto const dtype =
    cudf::data_type{std::is_signed_v<T> ? cudf::type_id::INT64 : cudf::type_id::UINT64};
  std::vector<std::vector<T>> cases{{T{100}, T{200}},
                                    {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()}};
  if constexpr (std::is_signed_v<T>) {
    cases.push_back({std::numeric_limits<T>::min(), std::numeric_limits<T>::min()});
    cases.push_back({std::numeric_limits<T>::max(), std::numeric_limits<T>::min()});
    cases.push_back({T{-100}, T{200}});
  }
  for (auto const& values : cases) {
    for (bool all_null : {false, true}) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(cudf::make_column_from_scalar(
        cudf::numeric_scalar<int32_t>{0, true, stream, mr}, values.size(), stream, mr));
      auto value_column = cudf::make_fixed_width_column(
        dtype,
        values.size(),
        all_null ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED,
        stream,
        mr);
      REQUIRE(cudaMemcpyAsync(value_column->mutable_view().template data<T>(),
                              values.data(),
                              values.size() * sizeof(T),
                              cudaMemcpyHostToDevice,
                              stream.value()) == cudaSuccess);
      columns.push_back(std::move(value_column));
      auto input =
        sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)), *space, stream);
      auto output = gpu_aggregate_impl::local_grouped_aggregate(
        input->to_read_only(),
        {0},
        {cudf::aggregation::Kind::SUM, cudf::aggregation::Kind::COUNT_VALID},
        {1, 1},
        {},
        {true, false},
        stream,
        *space);
      auto read_only = output->to_read_only();
      auto view      = sirius::get_cudf_table_view(read_only);
      REQUIRE(view.num_rows() == 1);
      REQUIRE(view.column(1).type().id() == cudf::type_id::DECIMAL128);
      REQUIRE(view.column(2).type().id() == cudf::type_id::INT64);
      int64_t count = -1;
      REQUIRE(cudaMemcpyAsync(&count,
                              view.column(2).template data<int64_t>(),
                              sizeof(count),
                              cudaMemcpyDeviceToHost,
                              stream.value()) == cudaSuccess);
      __int128 sum = 0;
      if (!all_null) {
        REQUIRE(cudaMemcpyAsync(&sum,
                                view.column(1).template data<__int128>(),
                                sizeof(sum),
                                cudaMemcpyDeviceToHost,
                                stream.value()) == cudaSuccess);
      }
      stream.synchronize();
      REQUIRE(count == (all_null ? 0 : static_cast<int64_t>(values.size())));
      if (all_null) {
        REQUIRE(view.column(1).null_count() == 1);
      } else {
        __int128 expected = 0;
        for (T value : values) {
          expected += static_cast<__int128>(value);
        }
        REQUIRE(sum == expected);
      }
    }
  }
}

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate grouped aggregates data_batch with single partition key, "
  "multiple aggregations on numerics",
  "[physical_grouped_aggregate]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 1000;

  // Create test data with single group key column
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 1, stream, mr);

  std::shared_ptr<data_batch> input_batch =
    sirius::make_data_batch(std::move(input_table), *space, stream);

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0},                      // group_indexes: GROUP BY column 0
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {1, 1, 1}                 // agg_indexes: MIN(column 1), MAX(column 1), COUNT(column 1)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs =
    grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());

  // Verify we got one output batch
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    true);
  REQUIRE(tables_match);
}

TEMPLATE_TEST_CASE("sirius_physical_grouped_aggregate grouped aggregates with AVG decomposition",
                   "[physical_grouped_aggregate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 100;

  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate_with_avg<Traits>(num_groups, 1, stream, mr);

  std::shared_ptr<data_batch> input_batch =
    sirius::make_data_batch(std::move(input_table), *space, stream);

  // AVG is decomposed into SUM + COUNT_VALID internally
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0},                             // GROUP BY column 0
    {"min", "max", "count", "avg"},  // aggregations including AVG
    {1, 1, 1, 1}                     // all on column 1
  );

  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs =
    grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // The local operator outputs expanded columns: group_key, min, max, count, sum, count_valid
  // (AVG decomposed into SUM + COUNT_VALID). Verify column count = 1 group + 5 aggregates.
  auto output_view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0]);
  REQUIRE(output_view.num_columns() == 6);  // 1 group + 3 non-avg + 2 (sum+count for avg)
}

TEMPLATE_TEST_CASE(
  "sirius_physical_grouped_aggregate grouped aggregates data_batch with multiple partition key, "
  "multiple aggregations",
  "[physical_grouped_aggregate]",
  int32_t,
  int64_t,
  float,
  double,
  int16_t,
  decimal64_tag,
  string_tag,
  timestamp_us_tag,
  date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto mr                = get_resource_ref(*space);
  auto stream            = default_stream();
  std::size_t num_groups = 1000;

  // Create test data with two group key columns
  auto [input_table, expected_table] =
    sirius::test::make_test_data_for_grouped_aggregate<Traits>(num_groups, 2, stream, mr);

  std::shared_ptr<data_batch> input_batch =
    sirius::make_data_batch(std::move(input_table), *space, stream);

  // Create aggregate expressions: GROUP BY column 0, SUM(column 1)
  auto agg_result = sirius::test::create_aggregate_expressions<Traits>(
    {0, 1},                   // group_indexes: GROUP BY column 0 and 1
    {"min", "max", "count"},  // aggregations: MIN, MAX, COUNT
    {2, 2, 2}                 // agg_indexes: MIN(column 2), MAX(column 2), COUNT(column 2)
  );

  // Create the grouped aggregate merge operator
  sirius_physical_grouped_aggregate grouped_aggregator(std::move(agg_result.output_types),
                                                       std::move(agg_result.aggregates),
                                                       std::move(agg_result.groups),
                                                       num_groups);

  auto outputs =
    grouped_aggregator.execute(pipelineable_operator_data({input_batch}), default_stream());

  // Verify we got one output batch
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches().size() == 1);

  // Compare output with expected using the validation utility
  // Sort both tables before comparison since aggregation order is not guaranteed
  bool tables_match = sirius::test::expect_data_batch_equivalent_to_table(
    dynamic_cast<const pipelineable_operator_data&>(*outputs).get_data_batches()[0],
    expected_table->view(),
    true);
  REQUIRE(tables_match);
}
