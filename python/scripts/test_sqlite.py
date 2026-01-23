#!/usr/bin/env -S uv run --quiet --script
"""
USearch SQLite Integration Tests

Test suite for USearch SQLite integration functionality,
testing database operations and vector search within SQLite.

Usage:
    uv run python/scripts/test_sqlite.py
    
Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy", 
#   "usearch"
# ]
# ///

import sqlite3
import json
import math

import numpy as np
import pytest

import usearch


try:
    found_sqlite_path = usearch.sqlite_path()
except FileNotFoundError:
    found_sqlite_path = None

if found_sqlite_path is None:
    pytest.skip(reason="Can't find an SQLite installation", allow_module_level=True)


batch_sizes = [1, 3, 20]
dimensions = [3, 97, 256]


def test_sqlite_minimal_json_cosine_vector_search():
    """Minimal test for searching JSON vectors in an SQLite database."""
    conn = sqlite3.connect(":memory:")

    # Loading extensions isn't supported in some SQLite builds,
    # including the default one on MacOS
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        pytest.skip("SQLite extensions are not available on this platform")
        return

    conn.load_extension(usearch.sqlite_path())
    cursor = conn.cursor()

    # Create a table with a JSON column for vectors
    cursor.executescript(
        """
        CREATE TABLE vectors_table (
            id SERIAL PRIMARY KEY,
            vector JSON NOT NULL
        );
        INSERT INTO vectors_table (id, vector)
        VALUES 
            (42, '[1.0, 2.0, 3.0]'),
            (43, '[4.0, 5.0, 6.0]');
    """
    )
    # Compute the distances to [7.0, 8.0, 9.0] using
    # the `distance_cosine_f32` extension function
    cursor.execute(
        """
        SELECT  vt.id, 
                distance_cosine_f32(vt.vector, '[7.0, 8.0, 9.0]') AS distance
        FROM vectors_table AS vt;
    """
    )
    ids_and_distances = list(cursor.fetchall())
    assert [ids_and_distances[0][0], ids_and_distances[1][0]] == [42, 43]
    assert ids_and_distances[0][1] < 0.05 and ids_and_distances[1][1] < 0.002


def test_sqlite_minimal_text_search():
    """Minimal test for Unicode strings in an SQLite database."""
    conn = sqlite3.connect(":memory:")

    # Loading extensions isn't supported in some SQLite builds,
    # including the default one on MacOS
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        pytest.skip("SQLite extensions are not available on this platform")
        return

    conn.load_extension(usearch.sqlite_path())
    cursor = conn.cursor()

    # Create a table with a TEXT column for strings
    str42 = "école"  # 6 codepoints (runes), 7 bytes
    str43 = "école"  # 5 codepoints (runes), 6 bytes
    str44 = "écolé"  # 5 codepoints (runes), 7 bytes
    assert str42 != str43, "etter 'é' as a single character vs 'e' + '´' are not the same"

    # Inject the different strings into the table
    cursor.executescript(
        f"""
        CREATE TABLE strings_table (
            id SERIAL PRIMARY KEY,
            word TEXT NOT NULL
        );
        INSERT INTO strings_table (id, word)
        VALUES 
            (42, '{str42}'),
            (43, '{str43}');
    """
    )
    cursor.execute(
        f"""
        SELECT  st.id, 

                distance_levenshtein_bytes(st.word, '{str44}') AS levenshtein_bytes,
                distance_levenshtein_unicode(st.word, '{str44}') AS levenshtein_unicode,
                distance_hamming_bytes(st.word, '{str44}') AS hamming_bytes,
                distance_hamming_unicode(st.word, '{str44}') AS hamming_unicode,

                distance_levenshtein_bytes(st.word, '{str44}', 2) AS levenshtein_bytes_bounded,
                distance_levenshtein_unicode(st.word, '{str44}', 2) AS levenshtein_unicode_bounded,
                distance_hamming_bytes(st.word, '{str44}', 2) AS hamming_bytes_bounded,
                distance_hamming_unicode(st.word, '{str44}', 2) AS hamming_unicode_bounded
        FROM strings_table AS st;
    """
    )
    ids_and_distances = list(cursor.fetchall())
    assert ids_and_distances[0] == (42, 5, 3, 7, 6, 2, 2, 2, 2)
    assert ids_and_distances[1] == (43, 2, 1, 2, 1, 2, 1, 2, 1)


def test_sqlite_blob_bits_vector_search():
    """Minimal test for searching binary vectors in an SQLite database."""

    conn = sqlite3.connect(":memory:")

    # Loading extensions isn't supported in some SQLite builds,
    # including the default one on MacOS
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        pytest.skip("SQLite extensions are not available on this platform")
        return

    conn.load_extension(usearch.sqlite_path())
    cursor = conn.cursor()

    # Create a table with a BLOB column for binary vectors
    cursor.executescript(
        """
        CREATE TABLE binary_vectors (
            id SERIAL PRIMARY KEY,
            vector BLOB NOT NULL
        );
        INSERT INTO binary_vectors (id, vector)
        VALUES 
            (42, X'FFFFFF'), -- 111111111111111111111111 in binary
            (43, X'000000'); -- 000000000000000000000000 in binary
        """
    )

    # Compute the distances between binary vectors and a sample vector using
    # the `distance_hamming_binary` and `distance_jaccard_binary` extension functions
    cursor.execute(
        """
        SELECT  bv.id, 
                distance_hamming_binary(bv.vector, X'FFFF00') AS hamming_distance,
                distance_jaccard_binary(bv.vector, X'FFFF00') AS jaccard_distance
        FROM binary_vectors AS bv;
        """
    )

    ids_and_distances = list(cursor.fetchall())
    np.testing.assert_array_almost_equal(ids_and_distances[0], (42, 8.0, 1.0 / 3))
    np.testing.assert_array_almost_equal(ids_and_distances[1], (43, 16.0, 1.0))


@pytest.mark.parametrize("num_vectors", batch_sizes)
@pytest.mark.parametrize("ndim", dimensions)
def test_sqlite_distances_in_high_dimensions(num_vectors: int, ndim: int):
    """
    Test the computation of cosine distances in high-dimensional spaces with random vectors stored in an SQLite database.

    This function tests the accuracy and consistency of cosine distance calculations between vectors in different formats:
    - distance_cosine_f32(JSON, JSON)
    - distance_cosine_f32(BLOB, BLOB)
    - distance_cosine_f16(BLOB, BLOB)

    The vectors are stored and retrieved as JSON strings and as binary blobs (in both 32-bit and 16-bit precision formats).
    The function asserts that the cosine similarities computed from the different storage formats (JSON, f32 BLOB, f16 BLOB)
    are within a certain tolerance of each other, ensuring that the distance calculations are consistent across different data formats.

    Parameters:
        num_vectors (int): The number of random vectors to generate and test.
        ndim (int): The dimensionality of each vector.
    """

    conn = sqlite3.connect(":memory:")

    # Loading extensions isn't supported in some SQLite builds,
    # including the default one on MacOS
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        pytest.skip("SQLite extensions are not available on this platform")
        return

    conn.load_extension(usearch.sqlite_path())
    cursor = conn.cursor()

    # Create a table with additional columns for f32 and f16 BLOBs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_table (
            id INTEGER PRIMARY KEY,
            vector_json JSON,
            vector_f32 BLOB,
            vector_f16 BLOB
        )
    """
    )

    # Generate and insert random vectors
    vectors = []

    for i in range(num_vectors):
        # Generate a random 256-dimensional vector
        vector = np.random.rand(ndim)
        vectors.append(vector)

        # Convert the vector to f32 and f16
        vector_f32 = np.float32(vector)
        vector_f16 = np.float16(vector)

        # Insert the vector into the database as JSON and as BLOBs
        cursor.execute(
            """
            INSERT INTO vector_table (vector_json, vector_f32, vector_f16) VALUES (?, ?, ?)
        """,
            (json.dumps(vector.tolist()), vector_f32.tobytes(), vector_f16.tobytes()),
        )

    # Commit changes
    conn.commit()

    similarities = """
    SELECT 
        a.id AS id1,
        b.id AS id2,
        distance_cosine_f32(a.vector_json, b.vector_json) AS cosine_similarity_json,
        distance_cosine_f32(a.vector_f32, b.vector_f32) AS cosine_similarity_f32,
        distance_cosine_f16(a.vector_f16, b.vector_f16) AS cosine_similarity_f16
    FROM 
        vector_table AS a,
        vector_table AS b
    WHERE 
        a.id < b.id;
    """
    cursor.execute(similarities)

    for a, b, similarity_json, similarity_f32, similarity_f16 in cursor.fetchall():
        assert math.isclose(similarity_json, similarity_f32, abs_tol=0.1)
        assert math.isclose(similarity_json, similarity_f16, abs_tol=0.1)

    # Clean up
    cursor.close()
    conn.close()


@pytest.mark.parametrize("num_vectors", batch_sizes)
def test_sqlite_distances_in_low_dimensions(num_vectors: int):

    # Setup SQLite connection and enable extensions
    conn = sqlite3.connect(":memory:")

    # Loading extensions isn't supported in some SQLite builds,
    # including the default one on MacOS
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        pytest.skip("SQLite extensions are not available on this platform")
        return

    conn.load_extension(usearch.sqlite_path())
    cursor = conn.cursor()

    # Create a table for storing vectors and their descriptions
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_table (
            id INTEGER PRIMARY KEY,
            vector_d0 FLOAT,
            vector_d1 FLOAT,
            vector_d2 FLOAT,
            vector_d3 FLOAT,
            description TEXT
        )
        """
    )

    # Insert random vectors into the table
    for i in range(num_vectors):
        vector = np.random.rand(4)  # Generate a random 4-dimensional vector
        cursor.execute(
            """
            INSERT INTO vector_table (vector_d0, vector_d1, vector_d2, vector_d3) VALUES (?, ?, ?, ?)
            """,
            tuple(vector),
        )

    conn.commit()

    # Query to calculate pairwise distances between vectors
    cursor.execute(
        """
        SELECT 
            a.id AS id1,
            b.id AS id2,
            distance_cosine_f32(a.vector_d0, a.vector_d1, a.vector_d2, a.vector_d3, b.vector_d0, b.vector_d1, b.vector_d2, b.vector_d3) AS cosine_similarity_f32,
            distance_cosine_f16(a.vector_d0, a.vector_d1, a.vector_d2, a.vector_d3, b.vector_d0, b.vector_d1, b.vector_d2, b.vector_d3) AS cosine_similarity_f16,
            distance_haversine_meters(a.vector_d0, a.vector_d1, b.vector_d0, b.vector_d1) AS haversine_meters
        FROM 
            vector_table AS a,
            vector_table AS b
        WHERE 
            a.id < b.id
        """
    )

    # Validate the results of the distance computations
    for id1, id2, similarity_f32, similarity_f16, haversine_meters in cursor.fetchall():
        assert 0 <= similarity_f32 <= 1, "Cosine similarity (f32) must be between 0 and 1"
        assert 0 <= similarity_f16 <= 1, "Cosine similarity (f16) must be between 0 and 1"
        assert haversine_meters >= 0, "Haversine distance must be non-negative"

    # Clean up
    cursor.close()
    conn.close()
