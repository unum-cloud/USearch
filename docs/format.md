# USearch File Format

USearch v3 checkpoints are SafeGraphs files, the interchange format specified at [safegraphs/SafeGraphs](https://github.com/safegraphs/SafeGraphs), and that specification is the authoritative description of every byte.
This page records only how USearch maps its indexes onto it.

## Monolith

A `.safegraphs` file holds exactly one shard's graph: an 8-byte magic, a little-endian header length, a JSON header, and a data segment of Arrow-layout buffers, each starting on the header's `alignment` boundary.
A dense index writes three regular columns.
The `key` column holds one identity per node, `u32`, `u64` or `uuid` by the key's width.
The `value` column holds one fixed-width vector per node, its dtype the stored scalar's own token, `u1x8`, `i4x2` and `u4x2` naming the packed ones, and its shape the logical width.
The `neighbors` column holds one fixed-degree row of node indices per node, where any index at or past the node count is a sentinel.
Producer columns may follow, such as the dense `norms` column.
The header carries `alignment`, the `metric` token, and any provenance the caller adds.

## Producer Columns

Beside `key`, `value` and `neighbors`, a header may carry producer columns, each an object of the same form under its own name.
Each one starts on the `alignment` boundary past the columns before it, and the file's length counts it.
A reader takes the names it knows, after checking their dtype, shape and row count, and skips the rest.

| Name    | Writer      | Dtype                   | Shape | Row                                   |
| ------- | ----------- | ----------------------- | ----- | ------------------------------------- |
| `norms` | dense index | the dot product's dtype | 2     | a node's sum, then its sum of squares |

A dense index writes `norms` where it keeps norms, and a dense load that finds none, or one it does not take, derives them from the vectors.

## Shard Layout

The number of live shards picks the layout, whatever the path's extension says.

| Live shards | Saved to a path        | Saved to a buffer      |
| ----------- | ---------------------- | ---------------------- |
| one         | a monolith             | a monolith             |
| several     | parts, then a manifest | a manifest, then parts |

Each part is a monolith of one shard's graph with that shard's own node IDs, so no edge crosses parts.
A path's parts are named `<stem>-0000k-of-0000N.safegraphs`, and its manifest lists them by `path`, `bytes` and `nodes`, written last as the commit record.
A buffer's manifest lists them by `offset` from the buffer's start instead, ends at the first zero byte, and every part follows on the header's `alignment` boundary.
Several shards never pack into one monolith: only compaction merges graphs.

## Loading Checkpoints

A loader tells the two apart by content, never by the extension: the magic makes a monolith, anything else a manifest.

| Checkpoint | Shards opened   |
| ---------- | --------------- |
| a monolith | one             |
| N parts    | N, one per part |

Placed shards the parts leave empty stay empty, and shards past the placed ones open over the call's memory domains or devices in turn.

## Residencies

| Index           | Saves to     | Loads from   | Views on NVMe |
| --------------- | ------------ | ------------ | ------------- |
| Dense on RAM    | path, buffer | path, buffer | path          |
| Dense on GPUs   | path, buffer | path, buffer | path          |
| Basic on RAM    | path         | path         | path          |
| Basic on NVIDIA | path         | path         | path          |

Dense GPUs are NVIDIA, AMD and Apple devices; the basic modalities are sequence, shapes and geospatial.
Any residency loads what any other saved.

## Key Columns

An index writes its keys at its own width: `u32`, `u64` or `uuid`.
A reader takes a column as wide as its keys or narrower, and refuses a wider one.

| Stored keys | `u32` reader | `u64` reader | `uuid` reader |
| ----------- | ------------ | ------------ | ------------- |
| `u32`       | as stored    | widened      | widened       |
| `u64`       | refused      | as stored    | widened       |
| `uuid`      | refused      | refused      | as stored     |

An integer widens into a wider integer by zero-extension, and into a `uuid` big-endian in its trailing 8 bytes, the identity the bindings render.
The C interface keys every index by `uuid`, so it loads all three.

## Neighbor Columns

The format describes `u32` and `u64` node IDs alike, and a reader takes only its own width.

| Reader                 | `u32` IDs | `u64` IDs |
| ---------------------- | --------- | --------- |
| Compiled library       | loads     | refused   |
| Dense templates        | loads     | refused   |
| Basic with `u32` nodes | loads     | refused   |
| Basic with `u64` nodes | refused   | loads     |

The compiled library's refusal names the 64-bit IDs, since its shards hold 32-bit ones.

## Basic Tier

Indexes over arbitrary payloads, such as sequences, write the same three columns with `value` and `neighbors` irregular.
Prefix sums of `u64` delimit each node's codec bytes and neighbor IDs, and the codec names the element dtype the header records.
