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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdlib>
#include <string_view>

namespace sirius::op::scan::cache_hit_metrics {

inline bool cache_hit_trace_enabled()
{
  static bool const on = [] {
    auto const* value = std::getenv("SIRIUS_CACHE_HIT_TRACE");
    return value != nullptr && std::string_view(value) == "1";
  }();
  return on;
}

// Logical decoded VALUE bytes, not allocated GPU memory or compressed I/O bytes.
// Excluding string offsets makes this additive across partial-read fragments:
// concatenation neither duplicates terminal offsets nor changes their width.
// Null masks, page padding and generated virtual columns are accounted separately.
// STRING size inspection can synchronize the stream; enable only in diagnostic runs.
inline std::size_t decoded_table_bytes(cudf::table_view const& table,
                                       rmm::cuda_stream_view stream)
{
  std::size_t bytes = 0;
  for (auto const& column : table) {
    if (column.size() == 0) { continue; }
    if (column.type().id() == cudf::type_id::STRING) {
      bytes += static_cast<std::size_t>(cudf::strings_column_view{column}.chars_size(stream));
    } else if (cudf::is_fixed_width(column.type())) {
      bytes += static_cast<std::size_t>(column.size()) * cudf::size_of(column.type());
    }
  }
  return bytes;
}

}  // namespace sirius::op::scan::cache_hit_metrics
