# The purpose of this file is to provide Pythonic wrapper on top
# the native precompiled CPython module. It improves compatibility
# Python tooling, linters, and static analyzers. It also embeds JIT
# into the primary `Index` class, connecting USearch with Numba.
from math import sqrt
from typing import Any

from usearch.index import CompiledMetric, MetricKind, MetricSignature, ScalarKind


def jit(
    ndim: int,
    metric: MetricKind = MetricKind.Cos,
    dtype: ScalarKind = ScalarKind.F32,
) -> CompiledMetric | MetricKind:
    """JIT-compiles the metric for target hardware and number of dimensions.

    This can result in up-to 3x performance difference on very large vectors
    and very recent hardware, as the Python module is compiled with high
    compatibility in mind and avoids very fancy assembly instructions.

    Uses Numba `cfunc` functionality, annotating it with Numba `types` instead
    of `ctypes` to support half-precision.
    https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
    """
    assert isinstance(metric, MetricKind)
    assert isinstance(dtype, ScalarKind)

    from numba import carray, cfunc, types  # type: ignore[import-not-found]

    signature_i8args = types.float32(types.CPointer(types.int8), types.CPointer(types.int8))
    signature_f16args = types.float32(types.CPointer(types.float16), types.CPointer(types.float16))
    signature_f32args = types.float32(types.CPointer(types.float32), types.CPointer(types.float32))
    signature_f64args = types.float32(types.CPointer(types.float64), types.CPointer(types.float64))

    numba_supported_types = (
        ScalarKind.I8,
        # Half-precision is still unsupported
        # https://github.com/numba/numba/issues/4402
        # ScalarKind.F16: np.float16,
        ScalarKind.F32,
        ScalarKind.F64,
    )
    if dtype not in numba_supported_types:
        return metric

    scalar_kind_to_accumulator_type = {
        ScalarKind.I8: types.int32,
        ScalarKind.F16: types.float16,
        ScalarKind.F32: types.float32,
        ScalarKind.F64: types.float64,
    }
    accumulator = scalar_kind_to_accumulator_type[dtype]

    def numba_ip(a: Any, b: Any) -> Any:
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        ab = accumulator(0)
        for i in range(ndim):
            ab += a_array[i] * b_array[i]
        return types.float32(1 - ab)

    def numba_cos(a: Any, b: Any) -> Any:
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        ab = accumulator(0)
        a_sq = accumulator(0)
        b_sq = accumulator(0)
        for i in range(ndim):
            ab += a_array[i] * b_array[i]
            a_sq += a_array[i] * a_array[i]
            b_sq += b_array[i] * b_array[i]
        a_norm = sqrt(a_sq)
        b_norm = sqrt(b_sq)
        if a_norm == 0 and b_norm == 0:
            return types.float32(0)
        elif a_norm == 0 or b_norm == 0 or ab == 0:
            return types.float32(1)
        else:
            return types.float32(1 - ab / (a_norm * b_norm))

    def numba_l2sq(a: Any, b: Any) -> Any:
        a_array = carray(a, ndim)
        b_array = carray(b, ndim)
        ab_delta_sq = accumulator(0)
        for i in range(ndim):
            ab_delta_sq += (a_array[i] - b_array[i]) * (a_array[i] - b_array[i])
        return types.float32(ab_delta_sq)

    scalar_kind_to_signature = {
        ScalarKind.I8: signature_i8args,
        ScalarKind.F16: signature_f16args,
        ScalarKind.F32: signature_f32args,
        ScalarKind.F64: signature_f64args,
    }

    metric_kind_to_function = {
        MetricKind.IP: numba_ip,
        MetricKind.Cos: numba_cos,
        MetricKind.L2sq: numba_l2sq,
    }

    if dtype == ScalarKind.I8 and metric == MetricKind.IP:
        metric = MetricKind.Cos

    pointer = cfunc(scalar_kind_to_signature[dtype])(metric_kind_to_function[metric])
    return CompiledMetric(
        pointer=pointer.address,
        kind=metric,
        signature=MetricSignature.ArrayArray,
    )
