/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op::scan {

struct fixed_page_mask_segment {
  void const* data{nullptr};
  std::size_t input_offset{0};
  std::size_t output_offset{0};
  std::size_t num_rows{0};
  cudf::type_id type_id{cudf::type_id::EMPTY};
};

struct fixed_page_simple_predicate {
  cudf::type_id type_id{cudf::type_id::EMPTY};
  bool has_lower{false};
  bool lower_inclusive{true};
  bool has_upper{false};
  bool upper_inclusive{true};
  int64_t lower_i64{0};
  int64_t upper_i64{0};
  double lower_f64{0.0};
  double upper_f64{0.0};
};

std::unique_ptr<cudf::column> make_fixed_page_simple_filter_mask(
  std::vector<fixed_page_mask_segment> const& segments,
  fixed_page_simple_predicate predicate,
  std::size_t output_rows,
  rmm::cuda_stream_view stream);

}  // namespace sirius::op::scan
