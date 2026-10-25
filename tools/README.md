# Command-Line Tools

USearch comes with command-line tools for building, merging and placing multi-billion-scale indexes, on one machine or across large CPU and GPU clusters, leveraging MPI and InfiniBand.
`usearch` runs every verb in one process, and `usearch-mpi` runs the same verbs across an MPI world, where every CPU or GPU in the network constructs its own shard and saves them together as one distributed SafeGraphs dataset.
Like every driver, they call the library through its C interface alone, and the library itself never links MPI, so the same library still loads on a laptop.

## How the Tools Are Organized

A command names a modality and its verb, like `usearch dense insert`, or a verb over checkpoints of any modality, like `usearch slots`, and prints one line per step.

- `usearch.cpp` parses the command line and dispatches to the verb.
- `dense.cpp` implements the dense verbs.
- `checkpoints.cpp` implements the verbs over checkpoints of every modality, at the header and at the byte level.
- `world.hpp` declares the process group a verb runs on: its rank, its size, a barrier and a gather.
- `world_local.cpp` is a world of one process, which `usearch` links.
- `world_mpi.cpp` is the MPI world, which `usearch-mpi` links, and the only file including `<mpi.h>`.

## Verbs

The verbs are named after the C interface they call:

- `usearch dense insert <vectors> --save <address>` creates an index, inserts a slice of one RetriEval file on a GPU or in RAM, and saves it.
  Each row is keyed by the same row of `--keys`, or by its dataset row number without one, so slices built apart keep distinct keys when merged.
- `usearch dense absorb <base> <donors>... --save <address>` loads the base, then loads and absorbs its donors one at a time, keeping their graphs rather than re-inserting their vectors, and saves the merged index.
  Each donor reports its rows and absorb time, the rows the closing refinement changed and how many rows a search reaches.
- `usearch dense refine <address> --save <address>` re-selects every row's neighbors against the finished graph, pass after pass until one changes nothing or `--max-refines` passes ran.
- `usearch dense split <address> --parts N --save <manifest>` cuts one checkpoint into `N` graph-local parts and commits them as a manifest at `<manifest>`, the parts beside it.
  The library refuses every split today with `unsupported`, and the verb reports that status and exits.
- `usearch dense health <address>` views a checkpoint from storage and audits how many of its rows a search reaches.
- `usearch metadata <address>` prints a monolith's or a manifest's nodes, parts, metric and keys, then reads its modality's metadata, picked by the metric its header names.
- `usearch slots <drive>` maps a raw drive: every live file with its owner and extent, and every free run of slots.
- `usearch copy <from> <to>` copies a checkpoint byte for byte, then reads it back and compares.
  A drive as the source copies the live file starting at its byte base, and a drive as the destination refuses to overlap a live slot; a drive named without a byte base takes the first free run of slots large enough.

A RAM index lives in the machine's first memory domain, and a VRAM one on device 0, or on the first ordinal `USEARCH_DEVICES` lists.
Either way the index is one shard, so its monolith is one graph.

Absorbing donors one at a time peaks at the base plus one donor, and pays one closing refinement per donor.
One grouped `usearch_dense_absorb` naming all `k` donors refines in ⌈log₂ k⌉ rounds over every row instead, but holds every donor at once.

## Addresses

Every verb takes an address wherever it reads or writes a checkpoint, so a build writes straight into a slot of a raw drive:

- `/data/sift.safegraphs` is a file, from its byte 0.
- `/dev/nvme1n1@64GiB` is a file or a device from a byte base, in `B`, `KiB`, `MiB`, `GiB` or `TiB`, or in bytes when the number is bare.
- `spdk://0000:5e:00.0@64GiB` is namespace 1 of a local PCIe controller, read through SPDK, and `spdk://0000:5e:00.0/2@64GiB` is its namespace 2.
- `trtype:TCP adrfam:IPv4 traddr:10.0.0.7 trsvcid:4420 subnqn:nqn.2026-10.io.example:index ns:1 base:0` is any transport ID SPDK parses, like NVMe over Fabrics, with its `ns:` and `base:` tokens.

A file whose name ends like a byte base, as in `run@64GiB`, is addressed as `run@64GiB@0`.
Binding a drive to SPDK takes SPDK's own [setup script](https://spdk.io/doc/getting_started.html), and `slots` and `copy` reach only drives the kernel's `nvme` driver holds.

## Raw Drives and Slots

A raw drive holds files back to back, with no filesystem and no label, by one convention shared with UStore:

- A drive is cut into __slots__ of 1 GiB, 2³⁰ bytes, from byte 0.
  Every file or volume starts on a slot boundary and covers whole slots, and slot 0 holds no label.
- Mapping reads the first 4 KiB of each slot.
  A slot is live when it opens a SafeGraphs monolith whose magic matches, whose JSON header parses, and whose declared extent fits on the drive, or a UStore volume whose `USTOREV1` magic and header CRC-32C hold.
- A live file covers ⌈length ÷ 1 GiB⌉ slots, which mapping skips, and every other slot is free.
  Owners come from the headers: the SafeGraphs metadata, or the UStore database UUID.
- Deleting a file zeroes its first block.

A save writes as much as the index needs, and nothing tells that size before it writes, so `--save` naming a drive takes a byte base on a slot boundary, at a slot no live file covers:

```sh
usearch slots /dev/nvme1n1
usearch dense insert data/base.100M.u8bin --stored-dtype u8 --metric euclidean --save /dev/nvme1n1@96GiB
usearch copy /data/index/sift.safegraphs /dev/nvme1n1
usearch metadata /dev/nvme1n1@96GiB
```

## Building

The tools build on Linux, where `slots` and `copy` reach block devices through `O_DIRECT`, and `usearch-mpi` builds wherever CMake's `FindMPI` finds an MPI implementation with its C compiler wrappers, such as Open MPI or MPICH.

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USEARCH_BUILD_TOOLS=ON
cmake --build build_release --target usearch_tools usearch_tools_mpi --parallel
cmake --install build_release --component tools
```

Add `-D USEARCH_BUILD_CUDA=ON` for `--residency vram`, `-D USEARCH_BUILD_SPDK=ON` for `spdk://` addresses, and `-D USEARCH_DENSE_DTYPES=6` or more to store `u8`.

## Recipe

SIFT's 100M vectors build as ten slices of 10M on one GPU, merge into one graph in RAM, and land in the first free slots of a raw drive:

```sh
for k in $(seq 0 9); do
    usearch dense insert data/sift-100m/base.100M.u8bin --residency vram \
        --first-vector $((k * 10000000)) --rows 10000000 --stored-dtype u8 --metric euclidean \
        --save /scratch/sift100m-$k.safegraphs
done

usearch dense absorb /scratch/sift100m-0.safegraphs $(seq -f /scratch/sift100m-%g.safegraphs 1 9) \
    --save /scratch/sift100m.safegraphs
usearch dense refine /scratch/sift100m.safegraphs --save /scratch/sift100m-refined.safegraphs
usearch metadata /scratch/sift100m-refined.safegraphs
usearch copy /scratch/sift100m-refined.safegraphs /dev/nvme1n1
```

## Running Across a Cluster

Under `usearch-mpi`, `dense insert` is a distributed build, and every other verb refuses to run on more than one rank.
Every rank takes one GPU or one memory domain and indexes a contiguous share of the `--first-vector` and `--rows` slice of a single dataset file that all of them can read.
The shares follow from the world size alone, so the ranks never exchange vectors: rank `r` of `R` takes rows `⌊N·r/R⌋` up to `⌊N·(r+1)/R⌋` of the slice's `N`, and each row is keyed by the same row of `--keys`, or by its dataset row number without one.
Each rank saves its shard as one SafeGraphs monolith, named the way a sharded save names its parts, `<stem>-00001-of-00064.safegraphs` and onward, beside the manifest `--save` names.
Once every rank reports its part complete, rank 0 writes the manifest at the `--save` address beside them.
That manifest is the commit record: a run that fails anywhere leaves parts and no checkpoint, and a reader sees it as absent.

The result is indistinguishable from what `usearch_dense_save` writes for a sharded index, so `usearch_dense_load_ram`, `usearch_dense_load_vram` and `usearch_dense_view_nvme` all open it.
A view and a VRAM load keep one shard per part, while a RAM load wants one memory domain named per part, as parts landing in one domain share a shard with no links between their graphs.
Every query walks every shard, so fewer and larger shards serve faster; a rank per GPU is the starting point that later merges compact.

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
- `--first-vector` defaults to `0`: the first row of the slice.
- `--rows` defaults to every row past `--first-vector`: rows in the slice.
- `--keys` defaults to row numbers: a `.u64bin` or `.u128bin` of one little-endian integer per vector row, so a UUID's 128-bit integer form keys its row whole.
- `--stored-dtype` defaults to the file's: `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2` or `u1x8`, cast on ingest; an `.fbin` file needs one.
- `--metric` defaults to `angular`: `inner`, `angular`, `euclidean`, `hamming` or `jaccard`; the last two for `u1x8`.
- `--residency` defaults to `vram` when a GPU is listed, and `ram` otherwise.
- `--connectivity`, `--expansion-add` and `--expansion-refine` default to the library's derivations.
- `--max-refines` defaults to `0`: refinement passes after ingestion, stopping early once no neighbors change.
- `--batch` defaults to `65536`: rows per insertion, with two batches staged at once, so one is read while the other is indexed.
- `--threads` defaults to 7/8 of the memory domain's cores: executor threads, `ram` only.
- `USEARCH_DEVICES` is an environment variable, not a flag: comma-separated device ordinals the ranks of a node take in turn.

The other verbs take only what they move:

- `dense absorb` and `dense refine` take `--save`, `--expansion-refine` and `--threads`, and `dense refine` also `--max-refines`, defaulting to `1`.
- `dense split` takes `--parts` of at least `2` and `--save` naming the manifest, plus `--expansion-refine` and `--threads`.
- `dense health` takes `--threads`, defaulting to what the devices' queues allow.
- `metadata`, `slots` and `copy` take no flags.

Every flag also reads its default from a `RETRIEVAL_` variable named after it, as the benchmarks do, so `--rows` defaults to `RETRIEVAL_ROWS`.
A bad value prints one line from rank 0, like `--metric="cosine" names no metric`, and every rank exits with status 1.
