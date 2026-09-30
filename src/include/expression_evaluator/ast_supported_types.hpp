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

// sirius
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"  // sirius::type_id

// duckdb
#include <duckdb/common/types.hpp>

// standard library
#include <array>

// Internal header shared by expression-executor .cpp files that need to consult the static
// allow-lists of AST-compatible CAST target types and BOUND_FUNCTION names. Keeps DuckDB
// includes out of the public expression_evaluator.hpp surface.

namespace sirius {

/// CAST return types that are currently safe to lower into a cuDF AST.
///
/// HUGEINT and UHUGEINT are here because this engine already maps them to cuDF INT64 and
/// UINT64 -- the same types BIGINT and UBIGINT map to (see get_cudf_type, which documents the
/// 128->64 narrowing). A cast to HUGEINT is therefore a cast to INT64, and leaving it out sent
/// it down the materializing fallback instead. DuckDB reaches that path on its own: it rewrites
/// SUM(x + 1) into sum(x) + count(x) * CAST(1 AS HUGEINT), whose scalar operand the fallback
/// cannot take -- ClickBench q30 is ninety of those.
inline constexpr std::array<duckdb::LogicalTypeId, 5> supported_ast_cast_types{
  {duckdb::LogicalTypeId::UBIGINT,
   duckdb::LogicalTypeId::BIGINT,
   duckdb::LogicalTypeId::DOUBLE,
   duckdb::LogicalTypeId::HUGEINT,
   duckdb::LogicalTypeId::UHUGEINT}};

/// Sirius-typed mirror of supported_ast_cast_types — same set of CAST target
/// types as above, expressed via sirius::type_id for native AST consumers.
inline constexpr std::array<sirius::type_id, 5> supported_ast_cast_types_native{
  {sirius::type_id::UBIGINT,
   sirius::type_id::BIGINT,
   sirius::type_id::DOUBLE,
   sirius::type_id::HUGEINT,
   sirius::type_id::UHUGEINT}};

/// BOUND_FUNCTION names that are currently safe to lower into a cuDF AST.
inline constexpr std::array<function_id, 6> supported_ast_functions{function_id::add,
                                                                    function_id::sub,
                                                                    function_id::mul,
                                                                    function_id::div,
                                                                    function_id::int_div,
                                                                    function_id::mod};

}  // namespace sirius
