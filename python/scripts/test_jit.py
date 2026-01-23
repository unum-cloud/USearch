#!/usr/bin/env -S uv run --quiet --script
"""
USearch JIT Compilation Tests

Test suite for Just-In-Time compilation functionality in USearch,
testing custom compiled metrics and performance optimizations.

Usage:
    uv run python/scripts/test_jit.py

Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "pytest",
#   "numpy",
#   "usearch"
# ]
# ///

import pytest
import numpy as np

import usearch
from usearch.eval import random_vectors
from usearch.index import (
    Index,
    MetricKind,
    MetricSignature,
    CompiledMetric,
)

dimensions = [3, 97, 256]
batch_sizes = [1, 77]


@pytest.mark.parametrize("ndim", dimensions)
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_index_numba(ndim: int, batch_size: int):
    """
    Uses Numba to show how Python code can be JIT compiled and injected into USearch.
    Uses the dot-product distance with different function signatures as an example.

    ! Requires the `numba` package to work.
    """
    try:
        from numba import cfunc, types, carray
    except ImportError:
        pytest.skip("Numba is not installed.")
        return

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    signature_two_args = types.float32(
        types.CPointer(types.float32),
        types.CPointer(types.float32),
    )
    signature_three_args = types.float32(
        types.CPointer(types.float32),
        types.CPointer(types.float32),
        types.uint64,
    )

    @cfunc(signature_two_args)
    def python_inner_product_two_args(a, b):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        c = 0.0
        for i in range(ndim):
            c += a_array[i] * b_array[i]
        return 1 - c

    @cfunc(signature_three_args)
    def python_inner_product_three_args(a, b, ndim):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        c = 0.0
        for i in range(ndim):
            c += a_array[i] * b_array[i]
        return 1 - c

    functions = [
        python_inner_product_two_args,
        python_inner_product_three_args,
    ]
    signatures = [
        MetricSignature.ArrayArray,
        MetricSignature.ArrayArraySize,
    ]
    for function, signature in zip(functions, signatures):
        metric = CompiledMetric(
            pointer=function.address,
            kind=MetricKind.IP,
            signature=signature,
        )
        index = Index(ndim=ndim, metric=metric, dtype=np.float32)

        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim)

        index.add(keys, vectors)
        matches = index.search(vectors, 10, exact=True)
        assert len(matches) == batch_size

        matches_keys = [match[0].key for match in matches] if batch_size > 1 else [matches[0].key]
        assert all(matches_keys[i] == keys[i] for i in range(batch_size)), f"Received {matches_keys}"


@pytest.mark.parametrize("ndim", [20, 50])
@pytest.mark.parametrize("batch_size", [100])
def test_index_numba_negative(ndim: int, batch_size: int):
    """
    Uses Numba to validate the hypothesis, that HNSW can work fine with arbitrary
    symmetric similarity measures, and not only with distance metrics.
    For that we construct a function that only returns negative values, and scales
    them before returning.

    ! Requires the `numba` package to work.
    """
    try:
        from numba import cfunc, types, carray
    except ImportError:
        pytest.skip("Numba is not installed.")
        return

    # Showcases how to use Numba to JIT-compile similarity measures for USearch.
    # https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    signature_two_args = types.float32(
        types.CPointer(types.float32),
        types.CPointer(types.float32),
    )

    @cfunc(signature_two_args)
    def normal_cosine_distance(a, b):
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        a2, b2, ab = 0.0, 0.0, 0.0
        for i in range(ndim):
            a2 += a_array[i] * a_array[i]
            b2 += b_array[i] * b_array[i]
            ab += a_array[i] * b_array[i]
        return 1 - ab / (np.sqrt(a2) * np.sqrt(b2))

    @cfunc(signature_two_args)
    def translated_cosine_distance(a, b):
        return (normal_cosine_distance(a, b) - 3) * 2

    # Create 2 indices
    normal_metric = CompiledMetric(
        pointer=normal_cosine_distance.address,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArray,
    )
    translated_metric = CompiledMetric(
        pointer=translated_cosine_distance.address,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArray,
    )
    normal_index = Index(
        ndim=ndim,
        metric=normal_metric,
        dtype=np.float32,
        # TODO: support `seed=42` for reproducibility
    )
    translated_index = Index(
        ndim=ndim,
        metric=translated_metric,
        dtype=np.float32,
        # TODO: support `seed=42` for reproducibility
    )

    # Populate them with identical data
    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)
    normal_index.add(keys, vectors, threads=1)
    translated_index.add(keys, vectors, threads=1)

    # Make sure, we receive the same keys
    count_queries = 10
    normal_matches = normal_index.search(vectors, count_queries, exact=False)
    translated_matches = translated_index.search(vectors, count_queries, exact=False)

    for query in keys.tolist():
        normal_keys = [normal_matches[query][i].key for i in range(count_queries)]
        translated_keys = [translated_matches[query][i].key for i in range(count_queries)]
        assert normal_keys == translated_keys, f"Expected {normal_keys} == {translated_keys} for key {query}"


# Just one size for Cppyy to avoid redefining kernels in the global namespace
@pytest.mark.parametrize("ndim", dimensions[-1:])
@pytest.mark.parametrize("batch_size", batch_sizes[-1:])
def test_index_cppyy(ndim: int, batch_size: int):
    """
    Uses Cppyy to show how C++ code can be JIT compiled and injected into USearch.
    Uses the dot-product distance with different function signatures as an example.

    ! Requires the `cppyy` package to work.
    """
    try:
        import cppyy
        import cppyy.ll
    except ImportError:
        pytest.skip("cppyy is not installed.")
        return

    cppyy.cppdef(
        """
    float inner_product_two_args(float *a, float *b) {
        float result = 0;
    #pragma unroll
        for (size_t i = 0; i != ndim; ++i)
            result += a[i] * b[i];
        return 1 - result;
    }
    
    float inner_product_three_args(float *a, float *b, size_t n) {
        float result = 0;
        for (size_t i = 0; i != n; ++i)
            result += a[i] * b[i];
        return 1 - result;
    }
    """.replace(
            "ndim", str(ndim)
        )
    )

    functions = [
        cppyy.gbl.inner_product_two_args,
        cppyy.gbl.inner_product_three_args,
    ]
    signatures = [
        MetricSignature.ArrayArray,
        MetricSignature.ArrayArraySize,
    ]
    for function, signature in zip(functions, signatures):
        metric = CompiledMetric(
            pointer=cppyy.ll.addressof(function),
            kind=MetricKind.IP,
            signature=signature,
        )
        index = Index(ndim=ndim, metric=metric, dtype=np.float32)

        keys = np.arange(batch_size)
        vectors = random_vectors(count=batch_size, ndim=ndim, dtype=np.float32)

        index.add(keys, vectors)
        matches = index.search(vectors, 10, exact=True)
        assert len(matches) == batch_size

        matches_keys = [match[0].key for match in matches] if batch_size > 1 else [matches[0].key]
        assert all(matches_keys[i] == keys[i] for i in range(batch_size)), f"Received {matches_keys}"


@pytest.mark.parametrize("ndim", [8])
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_index_peachpy(ndim: int, batch_size: int):
    """
    Uses PeachPy to show how x86_64 assembly code can be JIT compiled and injected into USearch.
    For brevety, we only use the dot-product distance with two float arrays of fixed size (8 dimensions)
    as an example. 8 such values fit perfectly into a single YMM register on x86.

    ! Runs only on x86_64 CPUs with AVX and AVX2 support.
    ! Requires the `py-cpuinfo` package to check if AVX2 is supported.
    ! Requires the `peachpy` package to assemble the code.
    """
    try:
        import platform

        arch = platform.machine()
        if arch != "x86_64":
            pytest.skip("We only use PeachPy for 64-bit x86.")
            return

        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if "avx2" not in info.get("flags", []):
            pytest.skip("Current CPU doesn't support AVX2.")
            return
    except ImportError:
        pytest.skip("PeachPy tests require `py-cpuinfo` to check if AVX2 is supported.")
        return

    try:
        from peachpy import (
            Argument,
            ptr,
            float_,
            const_float_,
        )
        from peachpy.x86_64 import (
            abi,
            Function,
            uarch,
            isa,
            GeneralPurposeRegister64,
            LOAD,
            YMMRegister,
            VSUBPS,
            VADDPS,
            VHADDPS,
            VMOVUPS,
            VFMADD231PS,
            VPERM2F128,
            VXORPS,
            RETURN,
        )
    except ImportError:
        pytest.skip("PeachPy is not installed.")
        return

    a = Argument(ptr(const_float_), name="a")
    b = Argument(ptr(const_float_), name="b")

    with Function("InnerProduct", (a, b), float_, target=uarch.default + isa.avx + isa.avx2) as asm_function:
        # Request two 64-bit general-purpose registers for addresses
        reg_a, reg_b = GeneralPurposeRegister64(), GeneralPurposeRegister64()
        LOAD.ARGUMENT(reg_a, a)
        LOAD.ARGUMENT(reg_b, b)

        # Load the vectors
        ymm_a = YMMRegister()
        ymm_b = YMMRegister()
        VMOVUPS(ymm_a, [reg_a])
        VMOVUPS(ymm_b, [reg_b])

        # Prepare the accumulator
        ymm_c = YMMRegister()
        ymm_one = YMMRegister()
        VXORPS(ymm_c, ymm_c, ymm_c)
        VXORPS(ymm_one, ymm_one, ymm_one)

        # Accumulate A and B product into C
        VFMADD231PS(ymm_c, ymm_a, ymm_b)

        # Reduce the contents of a YMM register
        ymm_c_permuted = YMMRegister()
        VPERM2F128(ymm_c_permuted, ymm_c, ymm_c, 1)
        VADDPS(ymm_c, ymm_c, ymm_c_permuted)
        VHADDPS(ymm_c, ymm_c, ymm_c)
        VHADDPS(ymm_c, ymm_c, ymm_c)

        # Negate the values, to go from "similarity" to "distance"
        VSUBPS(ymm_c, ymm_one, ymm_c)

        # A common convention is to return floats in XMM registers
        RETURN(ymm_c.as_xmm)

    python_function = asm_function.finalize(abi.detect()).encode().load()
    metric = CompiledMetric(
        pointer=python_function.loader.code_address,
        kind=MetricKind.IP,
        signature=MetricSignature.ArrayArray,
    )
    index = Index(ndim=ndim, metric=metric, dtype=np.float32)

    keys = np.arange(batch_size)
    vectors = random_vectors(count=batch_size, ndim=ndim)

    index.add(keys, vectors)
    matches = index.search(vectors, 10, exact=True)
    assert len(matches) == batch_size

    matches_keys = [match[0].key for match in matches] if batch_size > 1 else [matches[0].key]
    assert all(matches_keys[i] == keys[i] for i in range(batch_size)), f"Received {matches_keys}"
