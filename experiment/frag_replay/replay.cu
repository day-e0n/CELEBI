// Replay a cache allocation trace on a cudaMallocAsync pool capped at the cache budget,
// and record how the budget is spent after every scan.
//
// usage: replay <trace> <budget_bytes> <out.csv>
//
// Trace: "A <id> <n> <bytes>... <payload>" accesses one cache entry (all of its buffers are
// allocated and freed together); "M" closes a scan and takes a measurement.
//
// Eviction mirrors the page store: LRU, triggered when the requested bytes would exceed the
// budget. When the requested bytes fit but the pool cannot place an allocation, that is
// external fragmentation: one more LRU entry is evicted and the allocation retried.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

struct entry {
  std::vector<void*> ptrs;
  std::size_t bytes{0};
  std::size_t payload{0};
  bool resident{false};
  bool reused{false};
  std::list<int>::iterator lru;
};

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: %s <trace> <budget_bytes> <out.csv> [resident.txt]\n", argv[0]);
    return 2;
  }
  std::size_t const budget = std::stoull(argv[2]);

  CHECK(cudaSetDevice(0));
  cudaMemPoolProps props{};
  props.allocType     = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id   = 0;
  props.maxSize       = budget;
  cudaMemPool_t pool;
  CHECK(cudaMemPoolCreate(&pool, &props));
  std::uint64_t keep = UINT64_MAX;  // never hand memory back to the driver, like a long-lived pool
  CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep));
  cudaStream_t st;
  CHECK(cudaStreamCreate(&st));

  std::unordered_map<int, entry> entries;
  std::list<int> lru;
  std::size_t live = 0, live_payload = 0, reused_payload = 0;
  std::size_t hit_bytes = 0, inserted_bytes = 0;
  std::size_t hits = 0, misses = 0, uncacheable = 0, frag_events = 0, frag_evicted = 0,
              giveups = 0, evicted_unreused = 0, evicted = 0;

  auto try_alloc = [&](std::size_t s) -> void* {
    void* p = nullptr;
    if (cudaMallocFromPoolAsync(&p, s, pool, st) != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
    return p;
  };
  auto evict_one = [&]() -> std::size_t {
    int const id = lru.front();
    lru.pop_front();
    auto& e = entries[id];
    for (void* p : e.ptrs) { CHECK(cudaFreeAsync(p, st)); }
    e.ptrs.clear();
    e.resident = false;
    live -= e.bytes;
    live_payload -= e.payload;
    if (e.reused) { reused_payload -= e.payload; } else { evicted_unreused += e.bytes; }
    evicted += e.bytes;
    return e.bytes;
  };

  std::FILE* out = std::fopen(argv[3], "w");
  // Optional: the ids of the entries resident at each measurement, one line per scan.
  std::FILE* resident = argc == 5 ? std::fopen(argv[4], "w") : nullptr;
  std::fprintf(out,
               "scan,budget,live_requested,live_payload,reused_payload,pool_used,pool_reserved,"
               "largest_free,hits,misses,uncacheable,frag_events,frag_evicted,giveups,evicted,"
               "evicted_unreused,served_bytes,inserted_bytes\n");

  std::ifstream in(argv[1]);
  std::string line;
  int scan = 0;
  while (std::getline(in, line)) {
    if (line.empty()) { continue; }
    if (line[0] == 'M') {
      CHECK(cudaStreamSynchronize(st));
      std::uint64_t used = 0, reserved = 0;
      CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
      CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
      // Largest single allocation the pool can still place, to 2MiB, by trying it.
      std::size_t const step = 2u << 20;
      std::size_t lo = 0, hi = (budget > used ? budget - used : 0) / step + 1;
      while (hi - lo > 1) {
        std::size_t const mid = (lo + hi) / 2;
        void* p               = try_alloc(mid * step);
        if (p != nullptr) {
          CHECK(cudaFreeAsync(p, st));
          CHECK(cudaStreamSynchronize(st));
          lo = mid;
        } else {
          hi = mid;
        }
      }
      std::fprintf(out, "%d,%zu,%zu,%zu,%zu,%llu,%llu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", scan,
                   budget, live, live_payload, reused_payload, (unsigned long long)used,
                   (unsigned long long)reserved, lo * step, hits, misses, uncacheable, frag_events,
                   frag_evicted, giveups, evicted, evicted_unreused, hit_bytes, inserted_bytes);
      if (resident != nullptr) {
        for (int id : lru) { std::fprintf(resident, "%d ", id); }
        std::fprintf(resident, "\n");
      }
      ++scan;
      continue;
    }
    std::istringstream ss(line.substr(2));
    int id, n;
    ss >> id >> n;
    std::vector<std::size_t> sizes(n);
    std::size_t total = 0;
    for (auto& s : sizes) { ss >> s; total += s; }
    std::size_t payload;
    ss >> payload;

    auto& e = entries[id];
    if (e.resident) {
      ++hits;
      hit_bytes += e.payload;
      lru.splice(lru.end(), lru, e.lru);
      if (!e.reused) { e.reused = true; reused_payload += e.payload; }
      continue;
    }
    ++misses;
    if (total > budget) { ++uncacheable; continue; }
    while (live + total > budget && !lru.empty()) { evict_one(); }
    bool placed = false;
    while (!placed) {
      std::vector<void*> got;
      for (auto s : sizes) {
        void* p = try_alloc(s);
        if (p == nullptr) { break; }
        got.push_back(p);
      }
      if (got.size() == sizes.size()) {
        e.ptrs = std::move(got);
        placed = true;
        break;
      }
      for (void* p : got) { CHECK(cudaFreeAsync(p, st)); }
      if (lru.empty()) { ++giveups; break; }
      ++frag_events;
      frag_evicted += evict_one();
    }
    if (!placed) { continue; }
    e.bytes = total;
    e.payload = payload;
    e.resident = true;
    e.reused = false;
    lru.push_back(id);
    e.lru = std::prev(lru.end());
    live += total;
    live_payload += payload;
    inserted_bytes += payload;
  }
  std::fclose(out);
  if (resident != nullptr) { std::fclose(resident); }
  std::printf("served_from_cache=%.2fGiB inserted=%.2fGiB ", hit_bytes / double(1 << 30),
              inserted_bytes / double(1 << 30));
  std::printf("hits=%zu misses=%zu uncacheable=%zu frag_events=%zu frag_evicted=%.2fGiB giveups=%zu "
              "evicted=%.2fGiB evicted_unreused=%.2fGiB\n",
              hits, misses, uncacheable, frag_events, frag_evicted / double(1 << 30), giveups,
              evicted / double(1 << 30), evicted_unreused / double(1 << 30));
  return 0;
}
