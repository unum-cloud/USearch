# Shapes Indexes

Similarity search over 3D point clouds of any size: molecules, conformers, binding pockets, protein backbones, scans.
Two entries never need the same point count, and no kind needs a correspondence between points, so a 20-atom ligand scores against a 60-atom one without padding, truncation or an atom mapping.
Every kind is invariant to rotation, reflection and translation, and symmetric bit for bit, which is what lets one distance weigh a graph edge.

## Residencies

- `ram_metric.hpp` + `ram_index.hpp`: the CPU baseline, NumKong shape kits per capability tier from serial through NEON, Haswell, Skylake, RVV and WASM, under the `basic::ram_index` graph with each node owning one packed cloud.
- `vram_nvidia_index.hpp` + `vram_nvidia_metric.cuh`: the NVIDIA sibling for Chamfer and Hausdorff, a block walking each query and a warp scoring each candidate.
- NVMe: a saved checkpoint is served in place by `usearch_shapes_view_nvme`, through `basic/nvme_index.hpp`, answering every read verb a RAM handle does.

Checkpoints are SafeGraphs monoliths keeping each cloud in its inserted pose, so distances after a reload agree with those before it to the storage precision.

## Scoring Kinds

Each shape is centred and turned onto its principal axes once, when stored; every kind then starts from the eight sign flips of those axes.

- `chamfer`: the mean distance from each point to its nearest partner, both directions; the cheapest, and lenient to a stray point.
- `hausdorff`: the worst such distance; the same cost, and one stray point decides.
- `wasserstein_procrustes`: Kabsch without a known correspondence, alternating an exact transport plan with the best rotation or reflection; the default for conformers and pockets.
- `gromov_wasserstein`: compares each side's internal distances and never the coordinates; the best ranker and the costliest, sized for molecules rather than thousands of points.

Store points as `f16` to halve memory, or `f32` for clouds spanning hundreds of ångströms, where `f16` steps grow past a tenth of one.
Set `points` only when every entry has one count, like a fixed-length backbone or a landmark set.

## Construction & Search

Every batch is a tape: all clouds' `x, y, z` triples end to end, and `count + 1` offsets counting points around them.
Keys need not be unique, so the conformers of one molecule share its key under a `values_per_key` bound:

```c
#include <usearch/shapes_index.h>

float const points[] = {
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,       // first conformer, 4 atoms
    0, 0, 0, 1, 0, 0, 0, 1.2f, 0, 0, 0, 0.9f, // second conformer of the same molecule
    0, 0, 0, 1, 1, 0, 2, 3, 0, 4, 5, 1,       // another molecule
};
size_t const offsets[] = {0, 4, 8, 12};
usearch_key_t const keys[] = {molecule, molecule, other_molecule};

usearch_shapes_index_t index = NULL;
usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {.index = &index,
                                                      .outcome = &outcome,
                                                      .metric = usearch_shapes_metric_wasserstein_procrustes_k,
                                                      .quantization = usearch_scalar_f16_k,
                                                      .values_per_key = 64});
usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .count = 3,
                                                  .keys = {keys, sizeof(usearch_key_t)},
                                                  .shapes = {points, offsets, usearch_scalar_f32_k}});
usearch_shapes_refine(&(usearch_shapes_refine_t) {.index = index, .executor = executor, .outcome = &outcome});

usearch_matches_view_t matches = {0};
usearch_shapes_search(&(usearch_shapes_search_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .count = 1,
                                                  .queries = {points, offsets, usearch_scalar_f32_k},
                                                  .matches = &matches,
                                                  .arena = arena,
                                                  .wanted = 10});
```

Insert cheaply and spend the wide `expansion_factor` on `usearch_shapes_refine`, which pays most here, where one comparison scores a whole shape against another.
A refinement is cancellable and resumes where it stopped, so it can run under a time budget while searches continue.

## Moving From Other Tools

Overlay screeners like ROCS and FastROCS score Gaussian volume overlap after a rigid alignment and report a shape Tanimoto, often combined with a color Tanimoto over pharmacophore features.
Moving such a workflow over changes four things:

- __Conformers.__
  Keep the multi-conformer generator you have, whether OMEGA, RDKit ETKDG or CREST, and insert every conformer under its molecule's key; `usearch_shapes_count` reports how many a key holds.
  Pick heavy atoms only or all atoms once, and use the same choice for queries and database.
- __Scores.__
  Results are distances, lower is better and zero is identical, not similarities in $[0, 1]$.
  Re-derive score cutoffs on a validation set instead of translating Tanimoto thresholds.
- __Chemistry.__
  Points carry no atom types or radii, so there is no color term; geometry alone ranks.
  Keep the color or pharmacophore score as a re-ranking step over the top hits.
- __Poses and mirror images.__
  Search returns keys and distances, not superposed poses; re-align the few hits you inspect with your existing overlay tool.
  Reflection invariance means enantiomers score as equals, unlike rotation-only overlays, so filter stereochemistry after the search when it matters.

Descriptor methods like USR and USRCAT reduce a conformer to 12 or 60 moment statistics before comparing.
Those fixed-length vectors belong to the dense modality, while this one keeps every point and compares the clouds themselves.

RMSD after Kabsch alignment, as in RDKit's `GetBestRMS` or PyMOL's `align`, needs an atom mapping and equal counts.
`wasserstein_procrustes` needs neither, never exceeds the Kabsch RMSD of a known atom order on equal counts, and equals it where that order is an optimal matching, so it can stand in as an index-friendly RMSD.
