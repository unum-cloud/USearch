# Baseline Indexes

The readable reference tier: a graph index over arbitrary payloads, scored by any batched metric you write, under keys you choose.
It needs the standard library and the OS headers only, so together with `utilities/` it ships as a standalone header-only library.
The sequence and geospatial modalities are instantiations of it, `sequence::ram_index` over string views and `geospatial::ram_index` over Earth-fixed positions.
Specialized optimizations, like packed state words, growth schedules, lock-free readers and SIMD kernel kits, belong to the dense family instead.

## Metrics

A metric is a builder that scores one round of candidates against one query at a time, so an expensive metric can batch, pack or upload its candidates before scoring them all at once:

- `reset(query)` opens a round for one query; a walk reopens the same query every round, so a metric that prepares its query keeps that work when nothing changed.
- `push(value)` stages one candidate and answers `staged_k`, or `refused_k` once its staging is full, which makes the walk drain before staging more.
- `drain(distances, radius)` scores everything staged, writes the distances in push order, and clears the staging.

A distance within `radius` must be exact, while one above it may be any value above it, so a kernel can stop scoring a candidate the moment it cannot come back under.
Every verb runs with no locks held, so a `drain` may take milliseconds without stalling another thread, and the values a round pushed stay valid until its `drain` returns.
A metric marked `is_transparent` accepts query forms other than the stored value, such as a `std::string_view` for `std::string` payloads.

```cpp
#include <cmath>  // `std::fabs`
#include <span>   // `std::span`
#include <vector> // `std::vector`

#include <usearch/basic/ram_index.hpp>

namespace us = unum::usearch;

/** Manhattan distance over vectors of any length, scored one staged round at a time. */
struct manhattan_metric_t {
    std::vector<float> const* query_ = nullptr;
    std::vector<std::vector<float> const*> staged_;

    void reset(std::vector<float> const& query) noexcept { query_ = &query, staged_.clear(); }
    us::basic::metric_staging_t push(std::vector<float> const& value) noexcept {
        return staged_.push_back(&value), us::basic::metric_staging_t::staged_k;
    }
    void drain(std::span<double> distances, double) noexcept {
        for (std::size_t i = 0; i != staged_.size(); ++i) {
            double sum = 0;
            for (std::size_t j = 0; j != query_->size(); ++j) sum += std::fabs((*query_)[j] - (*staged_[i])[j]);
            distances[i] = sum;
        }
        staged_.clear();
    }
};
```

## Construction & Search

An index is a template over the payload, the key, the node ID width and the distance type, and it allocates nothing until `reserve` freezes its capacity.
Each thread brings its own metric instance and its own traversal context, which holds the walk's scratch and receives the matches:

```cpp
using index_t = us::basic::ram_index<std::vector<float>, us::utilities::u64_t>;
index_t index;
if (!index.reserve(1000)) return 1;

manhattan_metric_t metric;
index_t::traversal_context_t context {/* seed */ 42};
std::vector<std::vector<float>> values = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
std::vector<us::utilities::u64_t> keys = {10, 20, 30};
auto const [inserted, insert_status] = index.insert(std::span {values}, std::span<us::utilities::u64_t const> {keys},
                                                    metric, std::span<us::utilities::u64_t> {}, context,
                                                    us::basic::keep_diverse_t {16});

std::vector<float> const query = {0, 0, 0.9f};
auto const [found, search_status] = index.search(query, metric, context);
for (std::size_t i = 0; i != found.count; ++i) {
    auto const [key, key_status] = index.key_of(context.results[i].node_id);
    // `context.results[i].distance` is the match's score, ascending.
}
```

- `insert` moves a batch of payloads in under their keys and is all-or-nothing: capacity and scratch for the whole batch are secured before the first value moves.
- The selector decides which candidates become neighbors, and carries the neighbors-per-node budget: `basic::keep_diverse_t` prunes by diversity, `basic::keep_closest_t` keeps the nearest.
  Both are per-call values, so heuristics and budgets can change between batches.
- `search` leaves its matches in `context.results`, nearest first, and `key_of`, `value_of`, `find` and `count` translate between node IDs and keys.
- A search may score with a different metric than the one the graph was built with, and a `fold` passed to `search` reads every scored round, for walks that reduce to something other than the nearest matches.

## Concurrency

The leading template parameter of `insert` and `search` is the caller's synchronization contract:

- `concurrent_k`, the default, admits any mix of readers and writers.
- `immutable_k` certifies no live mutators, so searches read neighbor rows without locks.
- `exclusive_k` certifies sole ownership, compiling every lock out for single-threaded builds.

A violated certification answers `contended_k` instead of returning a silently wrong result.
`reserve`, `save`, `load` and `health` never run concurrently with inserts.

## Serialization

Payloads are arbitrary objects, so `save` and `load` take a codec that encodes each value into its own slice and back:

- `storage_requirements(value)` sizes one value's slice.
- `save(value, output)` encodes it.
- `load(input, value)` decodes it into a pre-constructed record.

`save` fills five caller-placed columns sized from `columns_extents`, while searches keep running, and `load` reads them back into a reserved, empty index, publishing everything at once.
