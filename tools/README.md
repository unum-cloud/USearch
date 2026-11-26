# Command-Line Tools

USearch comes with command-line tools for building, merging and placing multi-billion-scale indexes, on one machine or across large CPU and GPU clusters, leveraging MPI and InfiniBand.
`usearch` runs every verb in one process, and `usearch-mpi` runs the same verbs across an MPI world, where every CPU or GPU in the network constructs its own shard and saves them together as one distributed SafeGraphs dataset.
Like every driver, they call the library through its C interface alone, and the library itself never links MPI, so the same library still loads on a laptop.

## How the Tools Are Organized

A command names a modality and its verb, like `usearch dense insert`, or a verb over checkpoints of any modality, like `usearch copy`, and prints one line per step.

- `usearch.cpp` parses the command line and dispatches to the verb, through the flag tables of `arguments.hpp`.
- `dense.cpp` implements the dense verbs.
- `sequence.cpp`, `shapes.cpp` and `geospatial.cpp` read their modality's entries and queries, and run the checkpoint verbs `world.hpp` writes once for every modality.
- `checkpoints.cpp` implements the verbs over checkpoints of every modality, at the header and at the byte level.
- `arrow_checkpoints.hpp` turns a checkpoint's columns into Apache Arrow fields and back, and reads and writes Arrow files and streams on paths and pipes, over the codec in `arrow_ipc.hpp`.
- `world.hpp` declares the process group a verb runs on: its rank, its size, a barrier and a gather, and the insert, refine, reshard, health and search steps every modality shares.
- `world_local.cpp` is a world of one process, which `usearch` links.
- `world_mpi.cpp` is the MPI world, which `usearch-mpi` links, and the only file including `<mpi.h>`.

## Verbs

The verbs are named after the C interface they call:

- `usearch dense insert <vectors> --save <address>` creates an index, inserts a slice of one RetriEval file on a GPU or in RAM, and saves it.
  Each row is keyed by the same row of `--keys`, or by its dataset row number without one, so slices built apart keep distinct keys when merged.
  `--input` reads an Arrow file or stream in place of the RetriEval file, as [Arrow Pipes](#arrow-pipes) describes.
- `usearch dense refine <address> --save <address>` runs refinement rounds, each one a health check, a heal, a draw of hubs, a gather into neighborhoods, a restitch of every neighborhood and a reach walking the island entries, until one answers other than improved or `--max-refines` rounds ran.
  The rounds share one plan, so every health check starts from the same root, and a loaded NVMe checkpoint's round is one pass over its parts.
  Each round prints its neighborhoods and the widest, the links it added and dropped, the entries it changed, and the unreached and stranded islands it closed and opened with.
  With a `--connectivity` wider than the checkpoint's, the load keeps every node's stored neighbors and leaves the new slots empty, and a round gives an entry the diverse links that lead somewhere new, so an M=32 graph widens toward an M=64 one without a rebuild.
- `usearch dense reshard <first> <others>... --save <address>` loads every checkpoint and packs them all into the first in one call, keeping their graphs rather than re-inserting their vectors, and saves the packed index.
  A manifest stands for its parts, so `usearch dense reshard <manifest> --save <address>` packs a split checkpoint back into one, its first part the target.
  `usearch dense reshard <address>... --parts N --save <manifest>` cuts the checkpoints into `N` graph-local parts instead, every other one gathered into the first before the cut, and commits them as a manifest at `<manifest>`, the parts beside it.
  No walk links a moved entry in: graphs move whole, and every target closes with a round that heals what arrived or what the split left unreached; the verb reports the entries, the time, the entries those rounds changed and how many a search reaches.
- `usearch dense health <address>` views a checkpoint from storage and checks how many of its entries a search reaches, the depths its walk took, and the entries beyond the most populated depth.
- `usearch dense search <address> --query-vectors <path>` views a checkpoint from storage under `--residency nvme`, and loads it into RAM otherwise, then searches every row of a RetriEval file and prints each query's keys and distances, then the queries per second.
  `--traversal scan` scores every entry for the exact matches, and `--output` keeps the matches as Arrow instead of printing them.
  `--input` reads the queries from an Arrow file or stream, and `--index` names the checkpoint in place of the positional.
- `usearch dense get --index <address> --input <path> --keys-column <name> --output <path>` retrieves every vector stored under each key of an Arrow column, and writes one `key`, `vector` row per entry, in the scalar `--outputs-dtype` names or the stored one.
  It counts the keys the index does not hold on its status line, and refuses an input none of whose keys it holds, as no row then names the column's type.
- `usearch sequence`, `usearch shapes` and `usearch geospatial` take the same `insert`, `refine`, `reshard`, `health` and `search` verbs over texts, point clouds and observations, on one process.
  `usearch <modality> metadata <address>` is `usearch metadata <address>`.
- `usearch metadata <address>` prints a monolith's or a manifest's nodes, parts, metric and keys, then reads its modality's metadata, picked by the metric its header names.
  Ragged neighbors columns print their links per entry, and fixed ones their slots per entry, unused slots padded with sentinels.
  `usearch metadata -` reads the checkpoint off standard input.
- `usearch copy <from> <to>` copies a checkpoint byte for byte, then reads it back and compares.
  A drive as the source copies the live file starting at its byte base, and a drive as the destination refuses to overlap a live slot; a drive named without a byte base takes the first free run of slots large enough.
- `usearch export <address> <file.arrow>` writes a checkpoint of any modality as an Arrow IPC file, one record batch per part, each column's bytes as its buffers.
  Keys become the `arrow.uuid` extension or unsigned integers, regular columns fixed-size lists, ragged ones large lists over the prefix sums, and each field's metadata names its SafeGraphs dtype and shape.
- `usearch import <file.arrow> <address>` lays every record batch out as one part of a checkpoint in memory, loads it into RAM through the modality its metric names, and saves it at the address.
  `--metric` refuses a file recording another metric, and names one for a file recording none.
- `usearch recall <truth.arrow> <run.arrow> --wanted 1,10` compares two searches' `--output` files over the same queries, on any modality.
  An entry holding a query's own key at distance zero leaves both, a run's matches stop at the largest `--wanted`, and each depth d prints d-recall at the largest: the share of the true d nearest the run's matches reach, by key or within the true d-th distance, so ties at the cut count either way.
  It refuses queries that differ, and a truth shallower than the largest `--wanted`, so a self-search asks for one match more.

A RAM index lives in the machine's first memory domain, and a VRAM one on device 0, or on the first ordinal `USEARCH_DEVICES` lists.
Either way the index is one shard, so its monolith is one graph.

One reshard naming all `k` checkpoints runs one closing round over every entry, rather than one per checkpoint, and holds every checkpoint at once.

The first Ctrl-C, or a SIGTERM, stops a long command at its next barrier: `insert` and `search` between their batches, `refine` and `reshard` at the next barrier of their round, and `import` before it writes.
The command then prints `stopped in <phase>`, saves what stands where it saves, and exits with 130; a stopped `search` writes none of its outputs, and a stopped `import` writes nothing.
`copy` and `export` finish the file in hand and `health` runs to its end, then each exits with 130.
A second Ctrl-C ends the process at once.

## Addresses

Every verb takes an address wherever it reads or writes a checkpoint, so a build writes straight into a slot of a raw drive:

- `/data/sift.safegraphs` is a file, from its byte 0.
- `/dev/nvme1n1@64GiB` is a file or a device from a byte base, in `B`, `KiB`, `MiB`, `GiB` or `TiB`, or in bytes when the number is bare.
- `/dev/nvme1n1@64GiB+32GiB` also bounds the part to 32 GiB past its base, in the same units, and a writer refuses to grow it further.
- `spdk://0000:5e:00.0@64GiB` is namespace 1 of a local PCIe controller, read through SPDK, and `spdk://0000:5e:00.0/2@64GiB` is its namespace 2.
- `trtype:TCP adrfam:IPv4 traddr:10.0.0.7 trsvcid:4420 subnqn:nqn.2026-10.io.example:index ns:1 base:0` is any transport ID SPDK parses, like NVMe over Fabrics, with its `ns:` and `base:` tokens.

A file whose name ends like a byte base, as in `run@64GiB`, is addressed as `run@64GiB@0`.
Binding a drive to SPDK takes SPDK's own [setup script](https://spdk.io/doc/getting_started.html), and `copy` reaches only drives the kernel's `nvme` driver holds.

## Raw Drives and Slots

A raw drive holds files back to back, with no filesystem and no label, by one convention:

- A drive is cut into __slots__ of 1 GiB, 2³⁰ bytes, from byte 0.
  Every file or volume starts on a slot boundary and covers whole slots, and slot 0 holds no label.
- Mapping reads the first 4 KiB of each slot.
  A slot is live when it opens a SafeGraphs monolith whose magic matches, whose JSON header parses, and whose declared extent fits on the drive, and free when its first page is all zeros.
  Any other slot is foreign, skipped without being identified, and a save never overwrites it.
- A live file covers ⌈length ÷ 1 GiB⌉ slots, which mapping skips, and every other slot is free or foreign.
  Owners come from the SafeGraphs metadata.
- Deleting a file zeroes its first block.

[DiskSlots](https://github.com/unum-cloud/DiskSlots) maps, copies and frees the slots of every owner on a drive.

A save writes as much as the index needs, and nothing tells that size before it writes, so `--save` naming a drive takes a byte base on a slot boundary, at a slot no live file covers.
The save is then bounded to the free run that slot starts, as `/dev/nvme1n1@96GiB+412GiB` when the next live file sits at slot 508, so a save that outgrows the run fails instead of overwriting its neighbor.
A save to a drive stages each part in anonymous memory and writes it out direct, so it never reads the blocks it is about to overwrite.

```sh
usearch dense insert data/base.100M.u8bin --stored-dtype u8 --metric euclidean --save /dev/nvme1n1@96GiB
usearch copy /data/index/sift.safegraphs /dev/nvme1n1
usearch metadata /dev/nvme1n1@96GiB
```

## Building

The tools build on Linux, where `copy` reaches block devices, writing them `O_DIRECT` and exclusively, and `usearch-mpi` builds wherever CMake's `FindMPI` finds an MPI implementation with its C compiler wrappers, such as Open MPI or MPICH.

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USEARCH_BUILD_TOOLS=ON
cmake --build build_release --target usearch_tools usearch_tools_mpi --parallel
cmake --install build_release --component tools
```

Add `-D USEARCH_CUDA_ARCH=native` for `--residency vram`, `-D USEARCH_WITH_LINUX_SPDK=ON` for `spdk://` addresses, and `-D USEARCH_DENSE_DTYPES=6` or more to store `u8`.

## Recipe

SIFT's 100M vectors build as ten slices of 10M on one GPU, merge into one graph in RAM, and land in the first free slots of a raw drive:

```sh
for k in $(seq 0 9); do
    usearch dense insert data/sift-100m/base.100M.u8bin --residency vram \
        --first-entry $((k * 10000000)) --entries 10000000 --stored-dtype u8 --metric euclidean \
        --save /scratch/sift100m-$k.safegraphs
done

usearch dense reshard /scratch/sift100m-0.safegraphs $(seq -f /scratch/sift100m-%g.safegraphs 1 9) \
    --save /scratch/sift100m.safegraphs
usearch dense refine /scratch/sift100m.safegraphs --save /scratch/sift100m-refined.safegraphs
usearch metadata /scratch/sift100m-refined.safegraphs
usearch copy /scratch/sift100m-refined.safegraphs /dev/nvme1n1
```

## Evaluating Recall

An exact scan writes the truth, a walk writes the run, and `recall` judges one against the other through the same files, so no evaluation needs anything past the C interface:

```sh
usearch dense search graph.safegraphs --query-vectors queries.u1x8bin --query-keys queries.u128bin \
    --traversal scan --wanted 11 --output truth.arrow
usearch dense search graph.safegraphs --query-vectors queries.u1x8bin --query-keys queries.u128bin \
    --expansion-search 32 --wanted 11 --output run.arrow
usearch recall truth.arrow run.arrow --wanted 1,10
```

## Arrow Pipes

Vectors, keys, matches and checkpoints all cross a pipe, so object stores are reached through their own tools and USearch holds no network code:

```sh
aws s3 cp s3://bucket/base.arrows - | usearch dense insert --input - --keys-column id --save - | aws s3 cp - s3://bucket/index.safegraphs
usearch dense search --index /mnt/s3/index.safegraphs --input queries.arrow --wanted 10 --output - | aws s3 cp - s3://bucket/results.arrows
aws s3 cp s3://bucket/index.safegraphs - | usearch metadata -
usearch dense get --index index.safegraphs --input wanted.arrows --keys-column id --outputs-dtype f32 --output vectors.arrow
```

| Flag                      | Verbs                   | Meaning                                                                           |
| :------------------------ | :---------------------- | :-------------------------------------------------------------------------------- |
| `--input <path\|->`       | `insert` `search` `get` | An Arrow file or stream, or standard input                                        |
| `--vectors-column <name>` | `insert` `search`       | The rows, by default the only `fixed_size_list` or `numkong.u1x8` column          |
| `--keys-column <name>`    | `insert` `search` `get` | A `fixed_size_binary(16)`, `uint64` or `int64` column, by default the row numbers |
| `--index <address\|->`    | `search` `get`          | The checkpoint, or standard input                                                 |
| `--save -`                | `insert`                | Writes the checkpoint to standard output                                          |
| `--output <path\|->`      | `search` `get`          | Arrow to standard output for `-`                                                  |
| `--outputs-dtype <name>`  | `get`                   | The scalar the vectors convert to, by default the stored one                      |

Readers go by the first bytes rather than the extension: `ARROW1` opens a file and the stream continuation word a stream, either of them on a path or a pipe, and an input missing its end is refused instead of reading as a shorter one.
Parquet is refused by name, as the tools have no Parquet reader.
A writer takes the file framing for an `.arrow` path, and the stream framing for an `.arrows` path or `-`.
Input columns reach the library through the same `usearch_*_view_from_arrow` converters every caller uses, so one column with a null or another layout fails its batch, and integer keys widen into the low bytes of a key, as row numbers do.

A `-` is one pipe, so `usearch-mpi` refuses it on more than one rank, and one verb cannot read both `--input -` and `--index -`.
While a checkpoint or Arrow bytes take standard output, every status line goes to standard error.
A checkpoint on standard input loads into RAM alone, under `--residency ram`.

## Running Across a Cluster

Under `usearch-mpi`, `dense insert` is a distributed build, and every other verb refuses to run on more than one rank.
Every rank takes one GPU or one memory domain and indexes a contiguous share of the `--first-entry` and `--entries` slice of a single dataset file that all of them can read.
The shares follow from the world size alone, so the ranks never exchange vectors: rank `r` of `R` takes rows `⌊N·r/R⌋` up to `⌊N·(r+1)/R⌋` of the slice's `N`, and each row is keyed by the same row of `--keys`, or by its dataset row number without one.
Each rank saves its shard as one SafeGraphs monolith, named the way a sharded save names its parts, `<stem>-00001-of-00064.safegraphs` and onward, beside the manifest `--save` names.
Once every rank reports its part complete, rank 0 writes the manifest at the `--save` address beside them.
That manifest is the commit record: a run that fails anywhere leaves parts and no checkpoint, and a reader sees it as absent.

The result is indistinguishable from what `usearch_dense_save` writes for a sharded index, so `usearch_dense_load_ram`, `usearch_dense_load_vram` and `usearch_dense_view_nvme` all open it.
A view and a VRAM load keep one shard per part, while a RAM load wants one memory domain named per part, as parts loaded into one domain share a shard with no links between their graphs.
Every query walks every shard, so fewer and larger shards serve faster; a rank per GPU is the starting point that a later reshard gathers into fewer shards.

GPUs are assigned to the ranks of a node in turn.
A launcher that hands each task its own GPU, like Slurm's `--gpus-per-task 1`, leaves every rank its device 0, and the `USEARCH_DEVICES` environment variable lists the ordinals a node's ranks take otherwise.
Without a GPU, or with `--residency ram`, the ranks of a node spread over its memory domains, so a dual-socket machine wants two ranks.

On one machine, `mpirun` is enough:

```sh
mpirun -np 2 usearch-mpi dense insert data/sift-10m/base.10M.u8bin --metric euclidean \
    --save /scratch/sift/sift.safegraphs
```

On a Slurm cluster, `srun` starts one rank per GPU and hands their ranks to MPI through PMIx, while every node mounts the same parallel filesystem:

```sh
srun --nodes 16 --ntasks-per-node 8 --gpus-per-task 1 --mpi pmix \
    usearch-mpi dense insert /lustre/datasets/deep-1b/base.1B.fbin \
    --save /lustre/indexes/deep-1b/deep.safegraphs --stored-dtype bf16 --metric euclidean --max-refines 1
```

On Kubernetes, the Kubeflow MPI Operator starts the same image as a launcher and its workers, with the dataset and output on a shared volume:

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: usearch-dense
spec:
  slotsPerWorker: 8
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
            - name: launcher
              image: usearch-tools
              command: [mpirun, -np, "16", usearch-mpi, dense, insert, /data/base.1B.u8bin, --save, /data/index/base.safegraphs]
    Worker:
      replicas: 2
      template:
        spec:
          containers:
            - name: worker
              image: usearch-tools
              resources:
                limits:
                  nvidia.com/gpu: 8
              volumeMounts:
                - name: data
                  mountPath: /data
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: usearch-data
```

Object storage joins at the edges: parts are written on a filesystem that admits positional writes, like Lustre, GPFS, WEKA or VAST, then uploaded with the manifest last, so a bucket never shows a manifest before its parts.

## Flags

`dense insert` takes the vectors, a RetriEval file of `.fbin`, `.bf16bin`, `.f16bin`, `.i8bin`, `.u8bin` or `.u1x8bin` that every rank can read, and these flags:

- `--save` is required: the address to save to, or across ranks the manifest to commit, its parts written beside it.
- `--first-entry` defaults to `0`: the first row of the slice.
- `--entries` defaults to every row past `--first-entry`: rows in the slice.
- `--keys` defaults to row numbers: a `.u64bin` or `.u128bin` of one little-endian integer per vector row, so a UUID's 128-bit integer form keys its row whole.
  Every modality's `insert` takes it, one little-endian integer per entry.
- `--stored-dtype` defaults to the file's: `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2` or `u1x8`, cast on ingest; an `.fbin` file needs one.
- `--metric` defaults to `angular`: `inner`, `angular`, `euclidean`, `hamming` or `jaccard`; the last two for `u1x8`.
- `--residency` defaults to `vram` when a GPU is listed, and `ram` otherwise.
- `--connectivity`, `--expansion-add` and `--expansion-refine` default to the library's derivations.
- `--max-refines` defaults to `0`: refinement rounds after ingestion, stopping early once one answers other than improved.
- `--memory-limit` defaults to the library's derivation: the bytes a loaded NVMe checkpoint's refine holds for the one part it rewrites, like `40G`, a smaller one streaming that part's rounds from the drives; RAM and VRAM rounds size themselves from each core's cache.
- `--batch` defaults to `65536`: entries per insertion, with two batches staged at once, so one is read while the other is indexed.
- `--threads` defaults to what the memory domains or the devices' queues allow: executor threads.
- `--nvme-class` and `--nvme-ram` configure a view, as for `search` below, and an insert passes them by.
- `USEARCH_DEVICES` is an environment variable, not a flag: comma-separated device ordinals the ranks of a node take in turn.

Every verb over a checkpoint opens the same session, so it takes `--residency`, `--nvme-class`, `--threads` and `--nvme-ram` alike, and every verb that refines takes `--expansion-refine` and `--memory-limit`.
Beyond those, the other verbs take only what they move:

- `reshard` and `refine` take `--save` and `--connectivity`, defaulting to the checkpoint's and only ever widening it in RAM or VRAM, and `refine` also `--max-refines`, defaulting to `1`.
  A packing `reshard` widens its first part, the target, so the merge lands at the wider degree.
  Both default to `--residency ram`, and a residency that cannot refine refuses.
- `reshard` also takes `--parts` of at least `2`, cutting its checkpoints, and `--save` then names the manifest.
- `health` defaults to `--residency nvme`, viewing the checkpoint from storage.
- `search` takes `--wanted`, defaulting to `10`, `--expansion-search`, `--traversal`, `walk` by default or `scan`, `--batch`, and the queries in the format its modality's insert reads.
  A view also takes `--nvme-ram`, the bytes its key table may take, like `24G`: with it, key lookups and hints probe the table, and without it, the default, they refuse; the open line then counts the table's rows and bytes.
  `--matches` writes every query's top keys as a `.u128bin`: `u32` rows, `u32` columns, then one little-endian 128-bit key per column, the byte order `--keys` reads, a query matching fewer than `--wanted` repeating its last, then one `u32` count per row, so a query matching none hints none.
  `--hints` reads such a file, one row per query, and every walk starts from the entries under its row's keys ahead of its random draws, so one index's matches seed another's over the same keys; a walk takes as many keys as it makes draws, the connectivity rounded up to whole quads, a view needs a key table for them, and sequence and geospatial refuse them; the throughput line then counts the hint keys no entry holds.
  Dense indexes take hints on every residency but VRAM, and shapes indexes in RAM alone.
  `--output` writes the matches as Arrow in place of printing them: one row per query, its key as `query` and its matches as `matches`, a `LargeList<Struct<key, distance>>`, the keys as `arrow.uuid` and the distances as `float64`.
  A path ending in `.arrows` and `-`, standard output, take the stream framing, and any other path the file framing; under `-` every step prints to standard error, so standard output holds only Arrow.
  `--query-keys` names the queries in that file: a `.u128bin` of one key per query, as `--hints` reads them, defaulting to row numbers.
- `recall` takes `--wanted`, defaulting to `1,10`.
- `metadata` and `copy` take no flags.

The other modalities' `insert` takes the flags `dense insert` does but `--stored-dtype` and `--metric`, keys every entry by its number in the file unless `--keys` names a `.u64bin` or `.u128bin` of one key per entry, refused when its rows are not the file's entries, and reads its entries as the modality's benchmark does:

- `sequence insert <texts>` reads one entry per line, a carriage return before a newline dropped, under the required `--metric` `protein-alignment`, `dna-overlap` or `utf8-containment`; each scores by its own costs unless `--costs` names a file of 65,537 bytes, 256x256 int8 similarities between bytes then an int8 gap cost, which the library folds into classes, bytes sharing their rows and columns sharing one, and which `utf8-containment` refuses; the checkpoint records the folded costs and classes, so `sequence search` takes `--query-texts` alone, and `sequence reshard --parts` takes the `--costs` the index was built with and opens its parts under them.
- `shapes insert <points> --counts <path>` reads every cloud's points end to end, an `.fbin` or `.f16bin` of 3 columns, and the points per cloud, an `.ibin` of 1 column, under `--metric` `chamfer`, `hausdorff`, `gromov-wasserstein` or `wasserstein-procrustes`; `shapes search` takes `--query-points` and `--query-counts`.
- `geospatial insert <observations>` reads an `.f64bin` of eleven columns per observation, and builds without a terrain; `geospatial search` takes `--query-observations`, `--relation` `near`, `sees` or `seen`, and `--radius` in metres.

Every flag also reads its default from a `RETRIEVAL_` variable named after it, as the benchmarks do, so `--entries` defaults to `RETRIEVAL_ENTRIES`.
Byte counts take `K`, `M`, `G` or `T` and an optional `B`, so `24G` and RetriEval's `24GB` read alike.
A bad value prints one line from rank 0, like `--metric="cosine" names no metric`, and every rank exits with status 1.
