"""
Prepares the explorer's datasets, kept out of every runtime path.

`wikipedia` builds the dense index from Cohere's `wikipedia-22-12-simple-embeddings`, 486K paragraphs of 768 dimensions
and about 1.5 GB to download, through the Python binding, saving one SafeGraphs monolith. Every paragraph is an entry
under its article's key, so selecting one node brings up the whole article.

`molecules` writes a development set for the shapes modality from a local `usearch-molecules` shard: each conformer's
heavy atoms as points, every conformer of a molecule under that molecule's key. The file holds a `u32` count, that many
`u32` molecule ordinals, that many plus one `u32` point offsets, then `f32` points of x, y and z.

Both write a titles tape indexed by key - article titles or SMILES - as a `u32` count, that many plus one `u32` offsets,
then the UTF-8 bytes they span.

    uv run --with datasets --with numpy --with usearch python explorer/prepare.py wikipedia --output data/wikipedia-simple
    uv run --with pyarrow --with numpy python explorer/prepare.py molecules \\
        --source data/usearch-molecules-1m/0000000000-0001000000.3D.parquet --output data/usearch-molecules-explorer

Then serve the repository and open `explorer/index.html` with
`?modality=dense&index=../data/wikipedia-simple/index.safegraphs&titles=../data/wikipedia-simple/titles.tape`, or
`?dev=shapes&source=../data/usearch-molecules-explorer/conformers.bin&titles=../data/usearch-molecules-explorer/titles.tape`.
"""

import argparse
import pathlib
import struct

import numpy as np
import numpy.typing as npt

batch_rows = 16384


def write_titles(path: pathlib.Path, titles: list[str]) -> None:
    encoded = [title.encode("utf-8") for title in titles]
    offsets = np.zeros(len(encoded) + 1, dtype="<u4")
    offsets[1:] = np.cumsum([len(title) for title in encoded])
    with open(path, "wb") as tape:
        tape.write(struct.pack("<I", len(encoded)))
        tape.write(offsets.tobytes())
        tape.write(b"".join(encoded))


def prepare_wikipedia(output: pathlib.Path, limit: int) -> None:
    from datasets import load_dataset

    import usearch
    from usearch import DenseIndex, DenseMetric, Scalar

    paragraphs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
    if limit:
        paragraphs = paragraphs.select(range(min(limit, len(paragraphs))))

    # Article ordinals in order of first appearance, and the longest article, which bounds entries per key.
    ordinals: dict[int, int] = {}
    titles: list[str] = []
    keys = np.empty(len(paragraphs), dtype=np.uint64)
    for row, (wiki_id, title) in enumerate(zip(paragraphs["wiki_id"], paragraphs["title"], strict=True)):
        if wiki_id not in ordinals:
            ordinals[wiki_id] = len(titles)
            titles.append(title)
        keys[row] = ordinals[wiki_id]
    values_per_key = int(np.bincount(keys.astype(np.int64)).max())

    pool = usearch.Executor.for_ram()
    index = DenseIndex.init_ram(768, DenseMetric.ANGULAR, Scalar.BF16, connectivity=16, values_per_key=values_per_key)
    index.reserve(len(paragraphs))
    embeddings = paragraphs.with_format("numpy", columns=["emb"])
    for first in range(0, len(paragraphs), batch_rows):
        last = min(first + batch_rows, len(paragraphs))
        index.on(pool).insert(keys[first:last], np.asarray(embeddings[first:last]["emb"], dtype=np.float32))
        print(f"inserted {last:,} of {len(paragraphs):,} paragraphs", flush=True)
    index.on(pool).refine()

    output.mkdir(parents=True, exist_ok=True)
    index.on(pool).save(str(output / "index.safegraphs"))
    write_titles(output / "titles.tape", titles)
    print(f"saved {len(paragraphs):,} paragraphs under {len(titles):,} articles into {output}")


def prepare_molecules(source: pathlib.Path, output: pathlib.Path, molecules: int) -> None:
    import pyarrow.parquet as parquet

    columns = ["input_row", "smiles", "status", "n_heavy_atoms", "conformer_coords"]
    ordinals: dict[int, int] = {}
    titles: list[str] = []
    identities: list[int] = []
    shapes: list[npt.NDArray[np.float32]] = []
    rows = (
        values
        for batch in parquet.ParquetFile(source).iter_batches(batch_size=batch_rows, columns=columns)
        for values in zip(*(batch.column(name).to_pylist() for name in columns), strict=True)
    )
    for row, smiles, status, heavy_atoms, coordinates in rows:
        if status != 0 or not coordinates:
            continue
        if row not in ordinals:
            if len(titles) == molecules:
                break
            ordinals[row] = len(titles)
            titles.append(smiles)
        identities.append(ordinals[row])
        # Hydrogens follow the heavy atoms, so the leading points are the molecule's skeleton.
        shapes.append(np.asarray(coordinates, dtype="<f4").reshape(-1, 3)[:heavy_atoms])

    offsets = np.zeros(len(shapes) + 1, dtype="<u4")
    offsets[1:] = np.cumsum([len(points) for points in shapes])
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "conformers.bin", "wb") as file:
        file.write(struct.pack("<I", len(shapes)))
        file.write(np.asarray(identities, dtype="<u4").tobytes())
        file.write(offsets.tobytes())
        file.write(np.concatenate(shapes).tobytes())
    write_titles(output / "titles.tape", titles)
    print(f"saved {len(shapes):,} conformers of {len(titles):,} molecules into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    datasets = parser.add_subparsers(dest="dataset", required=True)
    wikipedia = datasets.add_parser("wikipedia")
    wikipedia.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/wikipedia-simple"))
    wikipedia.add_argument("--limit", type=int, default=0, help="paragraphs to keep, zero keeping all of them")
    molecules = datasets.add_parser("molecules")
    molecules.add_argument("--source", type=pathlib.Path, required=True)
    molecules.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/usearch-molecules-explorer"))
    molecules.add_argument("--molecules", type=int, default=2000)
    arguments = parser.parse_args()
    if arguments.dataset == "wikipedia":
        prepare_wikipedia(arguments.output, arguments.limit)
    else:
        prepare_molecules(arguments.source, arguments.output, arguments.molecules)


if __name__ == "__main__":
    main()
