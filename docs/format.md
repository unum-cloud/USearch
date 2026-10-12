# USearch File Format

USearch v3 checkpoints are SafeGraphs files, the interchange format specified at [safegraphs/SafeGraphs](https://github.com/safegraphs/SafeGraphs), and that specification is the authoritative description of every byte.
This page records only how USearch maps its indexes onto it.

## Monolith

A `.safegraphs` file holds one flat graph: an 8-byte magic, a little-endian header length, a JSON header, and a data segment of Arrow-layout buffers, each starting on the header's `alignment` boundary.
A dense index writes three regular columns.
The `key` column holds one identity per node, `u32`, `u64` or `uuid` by the key's width.
A reader takes keys no wider than its own and zero-extends them, an integer into a `uuid` big-endian in its trailing bytes, so the benchmarks' `u32` keys open in the C interface's `uuid` collections.
The `value` column holds one fixed-width vector per node, its dtype the stored scalar's own token, `u1x8`, `i4x2` and `u4x2` naming the packed ones, and its shape the logical width.
The `neighbors` column holds one fixed-degree row of `u32` node indices per node, where any index at or past the node count is a sentinel.
A `norms` column follows where the index keeps norms: a sum and a sum of squares per node, in the dot product's dtype.
A multi-index consolidates every shard into one such graph, rows in placement order and node IDs rebased, so the file reads as one index anywhere.
The header carries `alignment`, the `metric` token, and any provenance the caller adds.

## Distributed

A `.safegraphs.json` manifest catalogs one monolith per shard, named `<stem>-00001-of-0000S.safegraphs` beside it, each with shard-local node IDs so no edge crosses files.
The manifest is the only partition record.
A monolith carries none, and a loader recovers a consolidated monolith's shards from its row count and the shard count alone, since rotation fills shards in `divide_round_up` stripes.

## Basic Tier

Indexes over arbitrary payloads, such as sequences, write the same three columns with `value` and `neighbors` irregular.
Prefix sums of `i64` delimit each node's codec bytes and neighbor IDs, and the codec names the element dtype the header records.
