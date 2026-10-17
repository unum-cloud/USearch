# Distributed Index Construction

USearch comes with built-in indexing acceleration for large CPU and GPU clusters, leveraging MPI and InfiniBand to accelerate the construction of multi-billion-scale indexes.
You link the C library, initiate the MPI world, point it at a distributed filesystem, and every CPU or GPU in the network constructs its own shard, saving them together as one distributed SafeGraphs dataset.
The library itself never links MPI, so the same binary still loads on a laptop, and only the drivers in this directory do.

## How It Works

Every rank takes one GPU or one memory domain and indexes a contiguous slice of a single dataset file that all of them can read.
The slices follow from the world size alone, so the ranks never exchange vectors: rank `r` of `R` takes rows `⌊N·r/R⌋` up to `⌊N·(r+1)/R⌋`, and each row is keyed by the same row of `--keys`, or by its global row number without one.
Each rank saves its shard as one SafeGraphs monolith, named the way a sharded save names its parts, `<stem>-00001-of-00064.safegraphs` and onward.
Once every rank reports its part complete, rank 0 writes `<stem>.safegraphs.json` beside them.
That manifest is the commit record: a run that fails anywhere leaves parts and no checkpoint, and a reader sees it as absent.

The result is indistinguishable from what `usearch_dense_save` writes for a sharded index, so `usearch_dense_load_ram`, `usearch_dense_load_vram` and `usearch_dense_view_nvme` all open it.
A view and a VRAM load keep one shard per part, while a RAM load wants one memory domain named per part, as parts landing in one domain share a shard with no links between their graphs.
Every query walks every shard, so fewer and larger shards serve faster; a rank per GPU is the starting point that later merges compact.

GPUs are assigned to the ranks of a node in turn.
A launcher that hands each task its own GPU, like Slurm's `--gpus-per-task 1`, leaves every rank its device 0, and the `USEARCH_DEVICES` environment variable lists the ordinals a node's ranks take otherwise.
Without a GPU, or with `--residency ram`, the ranks of a node spread over its memory domains, so a dual-socket machine wants two ranks.

## Building

The drivers need an MPI implementation with its C compiler wrappers, such as Open MPI or MPICH, found by CMake's `FindMPI`.

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USEARCH_BUILD_DISTRIBUTED=ON
cmake --build build_release --target usearch_distributed_dense --parallel
```

## Running

On one machine, `mpirun` is enough:

```sh
mpirun -np 2 build_release/usearch_distributed_dense \
    --vectors data/sift-10m/base.10M.u8bin --output /tmp/sift/sift.safegraphs.json --metric euclidean
```

On a Slurm cluster, `srun` starts one rank per GPU and hands their ranks to MPI through PMIx, while every node mounts the same parallel filesystem:

```sh
srun --nodes 16 --ntasks-per-node 8 --gpus-per-task 1 --mpi pmix \
    usearch_distributed_dense \
    --vectors /lustre/datasets/deep-1b/base.1B.fbin --output /lustre/indexes/deep-1b/deep.safegraphs.json \
    --stored-dtype bf16 --metric euclidean --max-refines 1
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
              image: usearch-distributed
              command: [mpirun, -np, "16", usearch_distributed_dense, --vectors, /data/base.1B.u8bin, --output, /data/index/base.safegraphs.json]
    Worker:
      replicas: 2
      template:
        spec:
          containers:
            - name: worker
              image: usearch-distributed
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

- `--vectors` is required: base vectors in the RetriEval layout, one of `.fbin`, `.bf16bin`, `.f16bin`, `.i8bin`, `.u8bin`, `.u1x8bin`, readable from every rank.
- `--keys` defaults to row numbers: a `.u64bin` or `.u128bin` of one little-endian integer per vector row, in the RetriEval layout of the vectors, so a UUID's 128-bit integer form keys its row whole.
- `--output` is required: the `.safegraphs.json` manifest to commit, its parts written beside it.
- `--stored-dtype` defaults to the file's: `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2` or `u1x8`, cast on ingest; an `.fbin` file needs one.
- `--metric` defaults to `angular`: `inner`, `angular`, `euclidean`, `hamming` or `jaccard`; the last two for `u1x8`.
- `--residency` defaults to `vram` when a GPU is listed, and `ram` otherwise.
- `--connectivity` defaults to the library's derivation: the graph degree.
- `--expansion-add` defaults to the library's derivation: candidate pool while inserting.
- `--max-refines` defaults to `0`: refinement passes after ingestion, stopping early once no neighbors change.
- `--expansion-refine` defaults to the library's derivation: candidate pool while refining.
- `--batch` defaults to `65536`: rows per insertion, with two batches staged at once, so one is read while the other is indexed.
- `--threads` defaults to 7/8 of the memory domain's cores: executor threads, `ram` only.
- `USEARCH_DEVICES` is an environment variable, not a flag: comma-separated device ordinals the ranks of a node take in turn.

A bad value prints one line from rank 0, like `--metric="cosine" names no metric`, and every rank exits with status 1.
