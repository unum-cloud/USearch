# USearch Extensions for SQLite

USearch exposes the SIMD-accelerated distance functions for SQLite databases aimed to accelerate vector search and fuzzy string matching operations.
This includes:

- [x] Cosine and Euclidean distances for float vectors,
- [x] Jaccard and Hamming distances for binary vectors,
- [x] Levenshtein and Hamming distances for strings,
- [x] Haversine distance for geographical coordinates.

The SIMD-acceleration covers AVX2, most subsets of AVX512, ARM NEON, and Arm SVE instruction sets, more than most BLAS libraries.
The implementations are coming from [SimSIMD](https://github.com/ashvardanian/simsimd) and [StringZilla](https://github.com/ashvardanian/stringzilla).
They are most efficient when vectors are stored as BLOBs, but for broader compatibility can also handle JSONs, and even separate columns containing vector elements.

## Installation

USearch currently ships the SQLite extensions as part of the Python wheels, so they can be obtained from PyPI.

```sh
pip install usearch # brings sqlite extensions
```

## Quickstart

To add USearch extensions to your SQLite database, you can use the following commands:

```py
import sqlite3
import usearch

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
conn.load_extension(usearch.sqlite_path())
```

Afterwards, the following script should work fine.

```sql
-- Create a table with a JSON column for vectors
CREATE TABLE vectors_table (
    id INTEGER PRIMARY KEY,
    vector JSON NOT NULL
);

-- Insert two 3D vectors
INSERT INTO vectors_table (id, vector)
VALUES 
    (42, '[1.0, 2.0, 3.0]'),
    (43, '[4.0, 5.0, 6.0]');

-- Compute the distances to [7.0, 8.0, 9.0] using
-- the `distance_cosine_f32` extension function
SELECT 
    id, 
    distance_cosine_f32(vt.vector, '[7.0, 8.0, 9.0]') AS distance
FROM vectors_table AS vt;
```

### Functionality

#### Strings

String functions accept BLOBs and TEXTs, and return the distance as an integer.
This includes: 

- `distance_levenshtein_bytes`, 
- `distance_levenshtein_unicode`, 
- `distance_hamming_bytes`, 
- `distance_hamming_unicode`.

The Levenshtein distance reports the number of insertions, deletions, and substitutions required to transform one string into another, while the Hamming distance only reports the substitutions.
When applied to strings of different length, the Hamming distance evaluates the prefix of the longer string and adds the length difference to the result.
The `_bytes` variants compute the distance in the number of bytes, while the `_unicode` variants compute the distance in the number of Unicode code points, assuming the inputs are UTF-8 encoded.
Take a look at the following three similar looking, but distinct strings:

- `école` - 6 code points (runes), 7 bytes.
- `école` - 5 code points (runes), 6 bytes.
- `écolé` - 5 code points (runes), 7 bytes.

Those have three different letter "e" variants, including the letter "é" as a single character, the letter "e" plus the accent "´", as well as the plain letter "e".
This would result in different distances between those strings, depending on the chosen metric.

```sql
CREATE TABLE strings_table (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL
);

INSERT INTO strings_table (id, word)
VALUES 
    (42, 'école'),
    (43, 'école');

SELECT  
    st.id, 

    distance_levenshtein_bytes(st.word, 'écolé') AS lb,
    distance_levenshtein_unicode(st.word, 'écolé') AS lu,
    distance_hamming_bytes(st.word, 'écolé') AS hb,
    distance_hamming_unicode(st.word, 'écolé') AS hu,

    distance_levenshtein_bytes(st.word, 'écolé', 2) AS lbb,
    distance_levenshtein_unicode(st.word, 'écolé', 2) AS lub,
    distance_hamming_bytes(st.word, 'écolé', 2) AS hbb,
    distance_hamming_unicode(st.word, 'écolé', 2) AS hub

FROM strings_table AS st;
```

The last 4 columns will contain the bounded versions of the distance, which are faster to compute, thanks to the early stopping condition.
They might be handy, if you want to accelerate search for something like an "autocomplete" feature in a search bar.
The output will look like this:

| id  | lb  | lu  | hb  | hu  | lbb | lub | hbb | hub |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42  | 5   | 3   | 7   | 6   | 2   | 2   | 2   | 2   |
| 43  | 2   | 1   | 2   | 1   | 2   | 1   | 2   | 1   |

#### Bit Vectors

If using SQLite as a feature store, you might be dealing with single-bit vectors.

- `distance_hamming_binary` - the Hamming distance, meaning number of bits that differ between two vectors,
- `distance_jaccard_binary` - the Jaccard distance, meaning the number of bits that differ between two vectors divided by the number of bits that are set in at least one of the vectors.

Here is an example:

```sql
CREATE TABLE binary_vectors (
    id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL
);
INSERT INTO binary_vectors (id, vector)
VALUES 
    (42, X'FFFFFF'), -- 111111111111111111111111 in binary
    (43, X'000000'); -- 000000000000000000000000 in binary
SELECT  bv.id, 
        distance_hamming_binary(bv.vector, X'FFFF00') AS hamming_distance,
        distance_jaccard_binary(bv.vector, X'FFFF00') AS jaccard_distance
FROM binary_vectors AS bv;
```

#### Dense Vectors

Distance functions for dense vectors can be used on both BLOBs and JSONs.
Every name is structured as `distance_<metric>_<type>`, where 

- `<metric>` is the name of the metric, like `cosine`, `inner`, `sqeuclidean`, and `divergence`,
- `<type>` is the type of the vector elements, like `f64`, `f32`, `f16`, and `i8`.

The `cosine` metric is the cosine similarity, the `inner` metric is the inner (dot) product, the `sqeuclidean` metric is the squared Euclidean distance, and the `divergence` metric is the Jensen-Shannon divergence - the symmetric variant of the Kullback-Leibler divergence.

- `distance_sqeuclidean_f64`
- `distance_cosine_f64`
- `distance_inner_f64`
- `distance_divergence_f64`
- `distance_sqeuclidean_f32`
- `distance_cosine_f32`
- `distance_inner_f32`
- `distance_divergence_f32`
- `distance_sqeuclidean_f16`
- `distance_cosine_f16`
- `distance_inner_f16`
- `distance_divergence_f16`
- `distance_sqeuclidean_i8`
- `distance_cosine_i8`
- `distance_inner_i8`
- `distance_divergence_i8`

#### Geographical Coordinates

When dealing with geographical or other spherical coordinates, you can use:

- `distance_haversine_meters` - the Haversine distance on the sphere multiplied by the Earth's radius in meters.

---

Feel free to contribute more examples to this file 🤗
