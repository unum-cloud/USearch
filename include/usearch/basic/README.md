# Baseline Indexes

The readable reference tier: a graph index over arbitrary payloads, scored by any batched metric you write, under keys you choose.
Its RAM tier needs the standard library and the OS headers only, so together with `utilities/` it ships as a standalone header-only library.
The sequence, shapes and geospatial modalities are instantiations of it: `sequence::ram_index` over StringZilla strings, `shapes::ram_index` over NumKong shape packs, and `geospatial::ram_index` over Earth-fixed positions.
Specialized optimizations, like lock-free readers and SIMD kernel kits, belong to the dense family instead.

## Standalone Use

A single `basic::ram_index` is a complete index on its own.
It needs no SafeGraphs files, no plural of shards and no UUIDs.
Keys may be any hashable type, node IDs may be 32 or 64 bits wide, and every record keeps its own key.
The plurals in `ram_indexes.hpp` and `vram_nvidia_indexes.hpp` are what the compiled library builds on top.
Only they narrow the choice, to 32-bit shards whose rows a shard table locates in 8 bytes.

## Residencies

- `ram_index.hpp`: the in-memory index, with the metric contract, the selectors, the codec and the plans described below.
- `nvme_index.hpp`: the read-only index of one saved part on block storage, opened from its `persisted_columns_t` and walked through the same storage engines as the dense tier.
- `nvme_indexes.hpp`: the read-only view of a saved checkpoint, one part per shard; both walk their parts through `nvme_parts_catalog`, a component over spans of the media and parts each owns.
- `vram_nvidia_index.hpp` and `vram_nvidia_index.cuh`: the NVIDIA base every modality's face derives from, in plain C++, and the kernel body every modality's kernels run with the kit their ladder picks.
- `vram_index.cuh`: the vendor-neutral device bodies for the walk, the diversity selection, insertion and re-selection, which NVCC and HIP-Clang share.
  Only the NVIDIA host face exists so far, and AMD and Apple have no basic index yet.

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

An index is a template over the payload, the key, the node ID width and the distance type.
It allocates nothing until its first `reserve` or insert, then grows on demand in chunks of node records that never move, so `reserve` is only a hint that covers ahead.
Every record builds its value from the index's allocator by uses-allocator construction, so the values a caller inserts are best built from `index.get_allocator()` too.
Each thread brings its own metric instance and its own traversal context, which holds the walk's scratch, sits on any allocator such as a per-thread arena, and receives the matches:

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
- `plan_insert` spreads one batch over threads: it claims, covers and dispenses the whole ID range, each thread lands a disjoint slice through the plan's own `insert` with a context `reserve_traversal` sized, and `finish` publishes the batch at once.
  Nothing can fail past the dispense, so every position of a plan must land, and a refused fan-out lands on the calling thread.
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
`save`, `load`, `clear` and `health` never run concurrently with inserts, while `reserve` runs at any time.

## Serialization

Payloads are arbitrary objects, so `save` and `load` take a codec that encodes each value into its own slice and back:

- `storage_requirements(value)` sizes one value's slice.
- `save(value, output)` encodes it.
- `load(input, value)` decodes it into a pre-constructed record.

`save` fills five caller-placed columns sized from `columns_extents`, while searches keep running, and `load` reads them back into an empty index, publishing everything at once.

## Any Format

Basic indexes serialize to any format, not only SafeGraphs.
RAM indexes save and load through the five caller-placed `columns`, whatever file or buffer the caller places them in.
NVMe indexes read any `persisted_columns_t`: keys, values and neighbors, each a `persisted_column_t` of elements and, when its rows are ragged, their `prefix_sums_t`, located by byte ranges on one medium.
SafeGraphs is the format the compiled library ships, and `safegraphs_persisted_part` in `utilities/safegraphs_format.hpp` is the producer bundled for it.
The one requirement for NVMe is that the columns are uncompressed and addressable by row.
Arrow IPC files and raw-drive slots qualify.
Compressed row-group formats like Parquet do not, since every hop of a walk reads a few hundred bytes at an exact offset.

Take a deliberately odd format, `graph.odd`, that no other tool reads.
It opens with a 64-byte header of its own offsets, then stores the neighbor IDs, the keys and the value bytes, in that order.
It narrows both prefix-sum columns from `u64` to `u32`, halving them, since its graphs hold fewer than four billion links and payload bytes.
The RAM index of the example above fills the five columns, the tool writes them in its own order, and an NVMe view walks the file knowing only where each column landed:

```cpp
#include <usearch/basic/nvme_index.hpp>

using us::utilities::dtype_t;
float_vectors_codec_t codec; // the caller's codec from Serialization above

auto const [extents, extents_status] = index.columns_extents(codec);
std::vector<us::utilities::u64_t> keys(extents.nodes), neighbor_sums(extents.nodes + 1), value_sums(extents.nodes + 1);
std::vector<us::utilities::u64_t> neighbor_ids(extents.neighbors);
std::vector<std::byte> values(extents.value_bytes);
if (us::utilities::failed(index.save({keys, neighbor_sums, neighbor_ids, value_sums, values}, codec))) return 1;
// Write `graph.odd` with any I/O: the header, `neighbor_ids`, `neighbor_sums` cast to `u32`, `keys`,
// `values` and `value_sums` cast to `u32`, noting where each one starts.

using us::utilities::prefix_sums_t;
std::size_t const sums_bytes = (extents.nodes + 1) * 4;
us::utilities::persisted_columns_t persisted;
persisted.nodes = extents.nodes;
persisted.neighbors = {.dtype = dtype_t::u64_k,
                       .data = {neighbor_ids_at, neighbor_ids_at + neighbor_ids.size() * 8},
                       .offsets = prefix_sums_t {dtype_t::u32_k, {neighbor_sums_at, neighbor_sums_at + sums_bytes}}};
persisted.keys = {.dtype = dtype_t::u64_k, .data = {keys_at, keys_at + keys.size() * 8}, .row_width = 1, .row_bytes = 8};
persisted.values = {.dtype = dtype_t::u8_k,
                    .data = {values_at, values_at + values.size()},
                    .offsets = prefix_sums_t {dtype_t::u32_k, {value_sums_at, value_sums_at + sums_bytes}}};

us::basic::nvme_index<float_vectors_codec_t, us::utilities::u64_t, us::utilities::u64_t> view {
    us::utilities::forkunion_allocator_t {}};
if (us::utilities::failed(view.open("graph.odd", persisted))) return 1;
```

The index never learns the format: only where each column starts, how wide its elements are, and how its rows are found.

## Storage

`basic::nvme_index` serves one part from block storage, read-only, through the same storage walk as the dense tier, and `basic::nvme_indexes` serves the SafeGraphs checkpoints a RAM index saved, one part per shard.
A view keeps each part's column offsets and, for every ragged column, an Elias-Fano directory of its row starts at a few bits per row, so every row costs one covering read.
It reaches ForkUnion through `utilities/nvme_device.hpp` and SmashTable for the directories.

- `nvme_index::open(path, columns)` opens one part as its `persisted_columns_t` locates its rows.
- `nvme_indexes::view(path, paths, metric, executor)` opens a monolith or a manifest, building the directories on the executor's threads.
- `search(count, make_metric, query_of, executor, config, keys, distances, counts)` keeps one metric per row slot, built by the caller's factory on the slot's arena, and hands it every landed value as the codec's borrowed `view`.
- `count`, `contains` and `get` scan the key columns, and `get` reads the value rows of the keys it found.
