"""Type stubs for the `usearch.lib` extension.

Stub files are never executed, so PEP 695 `type` statements and PEP 688's `Buffer` are read
by any checker regardless of the interpreter running the tests. The package floor is 3.12.
"""

from typing import NoReturn, final

from usearch.dense_index import (
    DenseDescription as DenseDescription,
    DenseIndex as DenseIndex,
    DenseMetadata as DenseMetadata,
    DenseProducts as DenseProducts,
    MetricName as MetricName,
)
from usearch.types import (
    Arena as Arena,
    CancelledError as CancelledError,
    CapacityOverflowError as CapacityOverflowError,
    ConfigMismatchError as ConfigMismatchError,
    ContendedError as ContendedError,
    DeviceError as DeviceError,
    Error as Error,
    Executor as Executor,
    InputOutputError as InputOutputError,
    InternalError as InternalError,
    InvalidArgumentError as InvalidArgumentError,
    Key as Key,
    KeyLike as KeyLike,
    MalformedError as MalformedError,
    Match as Match,
    Matches as Matches,
    MoodName as MoodName,
    OutOfMemoryError as OutOfMemoryError,
    PathLike as PathLike,
    ReadOnlyError as ReadOnlyError,
    ScalarAlias as ScalarAlias,
    ScalarKind as ScalarKind,
    ScalarName as ScalarName,
    ShapeMismatchError as ShapeMismatchError,
    Stats as Stats,
    Tensor as Tensor,
    UninitializedError as UninitializedError,
    UnknownKindError as UnknownKindError,
    UnsupportedError as UnsupportedError,
    UnsupportedLayoutError as UnsupportedLayoutError,
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

@final
class SequenceIndex:
    """Similarity search over variable-length sequences."""

    def __init__(self) -> NoReturn: ...

@final
class MatrixIndex:
    """Similarity search over rank-2 tensors."""

    def __init__(self) -> NoReturn: ...

@final
class GeospatialIndex:
    """Similarity search over space and time."""

    def __init__(self) -> NoReturn: ...

# endregion: Other Modalities
