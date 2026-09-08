"""Type stubs for the `usearch.lib` extension.

Stub files are never executed, so PEP 695 `type` statements and PEP 688's `Buffer` are read
by any checker regardless of the interpreter running the tests. The package floor is 3.12.
"""

from typing import NoReturn

from usearch.dense_index import (
    DenseCursor as DenseCursor,
    DenseFinder as DenseFinder,
    DenseIndex as DenseIndex,
    DenseMetadata as DenseMetadata,
    MetricName as MetricName,
)
from usearch.types import (
    Error as Error,
    Executor as Executor,
    Key as Key,
    KeyLike as KeyLike,
    Match as Match,
    Matches as Matches,
    PathLike as PathLike,
    ScalarAlias as ScalarAlias,
    ScalarKind as ScalarKind,
    ScalarName as ScalarName,
    Stats as Stats,
    Tensor as Tensor,
)

# region: Library Probes

__version__: str

def version() -> str:
    """The library's semantic version."""
    ...

def capabilities_compiled() -> str:
    """The kernel families this build carries, whatever the machine offers."""
    ...

def capabilities_available() -> str:
    """The kernel families this machine can run, and dispatch picks from."""
    ...

# endregion: Library Probes

# region: Other Modalities

class SequenceIndex:
    """Similarity search over variable-length sequences."""

    def __init__(self, *args: object, **keywords: object) -> NoReturn: ...

class MatrixIndex:
    """Similarity search over rank-2 tensors."""

    def __init__(self, *args: object, **keywords: object) -> NoReturn: ...

class GeospatialIndex:
    """Similarity search over space and time."""

    def __init__(self, *args: object, **keywords: object) -> NoReturn: ...

# endregion: Other Modalities
