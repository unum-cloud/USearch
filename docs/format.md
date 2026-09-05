# USearch File Format

USearch v3 checkpoints are SafeGraphs files, the interchange format specified at [safegraphs/SafeGraphs](https://github.com/safegraphs/SafeGraphs), and that specification is the authoritative description of every byte.
This page records only how USearch maps its indexes onto it.

## Monolith

A `.safegraphs` file holds one flat graph: an 8-byte magic, a little-endian header length, a JSON header, and a data segment of Arrow-layout buffers, each starting on the header's `alignment` boundary.
A dense index writes three regular columns.
The `key` column holds one identity per node, `u32`, `u64` or `bin16` by the key's width.
The `value` column holds one fixed-width vector per node, with sub-byte scalars packed into `u8` rows and the logical width in `shape_effective`.
The `neighbors` column holds one fixed-degree row of `u32` node indices per node, where any index at or past the node count is a sentinel.
A multi-index consolidates every shard into one such graph, rows in placement order and node ids rebased, so the file reads as one index anywhere.
The header carries `nodes`, `alignment`, the `metric` token, the stored scalar as `quantization`, and any provenance the caller adds.

## Distributed

A `.safegraphs.json` manifest catalogs one monolith per shard, named `<stem>-00001-of-0000S.safegraphs` beside it, each with shard-local node ids so no edge crosses files.
The manifest is the only partition record.
A monolith carries none, and a loader recovers a consolidated monolith's shards from its row count and the shard count alone, since rotation fills shards in `divide_round_up` stripes.

## Basic Tier

Indexes over arbitrary payloads, such as sequences, write the same three columns with `value` and `neighbors` irregular.
Prefix sums of `i64` delimit each node's codec bytes and neighbor ids, and the codec names the element dtype the header records.
