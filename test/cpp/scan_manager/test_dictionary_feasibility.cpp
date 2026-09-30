/*
 * Feasibility probe for dictionary-encoding STRING columns in the page cache.
 *
 * The page cache stores DECODED columns, so a STRING column costs its characters
 * plus 4 bytes of offsets per row -- ClickBench's Title is 9.2 GB and URL 8.8 GB
 * decoded, against a per-entry admission cap of half the cache budget. They can
 * never be admitted, and that is why the four heaviest queries (77% of the scan
 * time the fixed-width cache cannot touch) get no cache at all.
 *
 * Dictionary encoding splits such a column into a fixed-width code per row plus
 * one copy of the distinct values. The codes are fixed-width, so they need no new
 * paging machinery -- they go through the existing fixed-width page path. This
 * measures whether that trade actually pays: how long the encode costs, how much
 * smaller the result is, and how far the per-chunk dictionaries diverge (each
 * chunk encodes independently, so serving a scan from several chunks needs one
 * unified key set).
 *
 * Not a correctness test; it is skipped unless the ClickBench parquet is present.
 */
#include "catch.hpp"

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr char const* kParquet = "/mnt/nvme/clickbench/hits_v2.parquet";

double ms_since(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::size_t strings_bytes(cudf::column_view const& col)
{
  cudf::strings_column_view scv(col);
  return scv.chars_size(cudf::get_default_stream()) +
         static_cast<std::size_t>(col.size()) * sizeof(std::int32_t);
}

}  // namespace

TEST_CASE("dictionary encoding shrinks ClickBench string columns", "[dictionary_probe][.]")
{
  if (!std::filesystem::exists(kParquet)) { SUCCEED("ClickBench parquet absent; skipping"); return; }

  for (auto const& column : {std::string{"SearchPhrase"}, std::string{"Title"}, std::string{"URL"}}) {
    std::size_t total_strings = 0, total_codes = 0, total_keys = 0, rows = 0;
    double read_ms = 0, encode_ms = 0;
    std::vector<std::unique_ptr<cudf::column>> dicts;

    for (int rg = 0; rg < 2; ++rg) {
      auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{kParquet})
                    .columns({column})
                    .row_groups({{rg}})
                    .build();
      auto const t_read = std::chrono::steady_clock::now();
      auto table        = cudf::io::read_parquet(opts);
      read_ms += ms_since(t_read);

      auto const view = table.tbl->view().column(0);
      rows += static_cast<std::size_t>(view.size());
      total_strings += strings_bytes(view);

      auto const t_enc = std::chrono::steady_clock::now();
      auto dict        = cudf::dictionary::encode(view);
      cudf::get_default_stream().synchronize();
      encode_ms += ms_since(t_enc);

      cudf::dictionary_column_view dcv(dict->view());
      total_codes += static_cast<std::size_t>(view.size()) * sizeof(std::int32_t);
      total_keys += strings_bytes(dcv.keys());
      dicts.push_back(std::move(dict));
    }

    auto const encoded = total_codes + total_keys;
    UNSCOPED_INFO("column=" << column);
    printf(
      "%-14s rows=%-10zu read=%7.0fms encode=%7.0fms | strings=%6.2f GB -> keys=%5.2f + codes=%5.2f"
      " = %5.2f GB (%4.1fx)\n",
      column.c_str(), rows, read_ms, encode_ms,
      total_strings / 1073741824.0, total_keys / 1073741824.0, total_codes / 1073741824.0,
      encoded / 1073741824.0, static_cast<double>(total_strings) / static_cast<double>(encoded));

    // Per-chunk dictionaries diverge; a cache entry spanning chunks needs one key
    // set. Measure what unifying them costs and how much the keys grow.
    std::vector<cudf::dictionary_column_view> views;
    views.reserve(dicts.size());
    for (auto const& d : dicts) { views.emplace_back(d->view()); }
    auto const t_match = std::chrono::steady_clock::now();
    auto matched       = cudf::dictionary::match_dictionaries(views);
    cudf::get_default_stream().synchronize();
    auto const match_ms = ms_since(t_match);

    std::size_t unified_keys = 0;
    if (!matched.empty()) {
      cudf::dictionary_column_view first(matched.front()->view());
      unified_keys = strings_bytes(first.keys());
    }
    printf("%-14s match_dictionaries=%6.0fms  per-chunk keys=%5.2f GB -> unified=%5.2f GB\n\n",
           column.c_str(), match_ms, total_keys / 1073741824.0, unified_keys / 1073741824.0);
    CHECK(encoded > 0);
  }
}
