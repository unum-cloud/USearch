Python API Reference
====================

The ``usearch`` package is one compiled extension, ``usearch.lib``, re-exported whole, plus three
pure-Python helpers that need NumPy.
``usearch/lib.pyi`` carries the typed signatures of everything the extension defines.

======
Dense
======

.. autoclass:: usearch.DenseIndex
    :members:
    :undoc-members:

.. autoclass:: usearch.On
    :members:
    :undoc-members:

.. autoclass:: usearch.DeferredOn
    :members:
    :undoc-members:

.. autoclass:: usearch.Deferred
    :members:
    :undoc-members:

.. autoclass:: usearch.DenseMetric
    :members:
    :undoc-members:

.. autoclass:: usearch.DenseMetadata
    :members:

.. autoclass:: usearch.DenseIndexDescription
    :members:

.. autoclass:: usearch.DenseReachability
    :members:

.. autoclass:: usearch.DenseTraversalStats
    :members:

.. autoclass:: usearch.DenseBatch
    :members:

.. autoclass:: usearch.DenseCursor
    :members:

.. autoclass:: usearch.DenseSearchStream
    :members:

==================
Executors & Arenas
==================

.. autoclass:: usearch.Executor
    :members:
    :undoc-members:

.. autoclass:: usearch.Arena
    :members:
    :undoc-members:

.. autoclass:: usearch.Matches
    :members:

.. autoclass:: usearch.Match
    :members:

.. autoclass:: usearch.DenseVectors
    :members:

.. autoclass:: usearch.DenseProducts
    :members:

.. autoclass:: usearch.Monolith
    :members:

===========
Vocabulary
===========

.. autofunction:: usearch.version

.. autofunction:: usearch.capabilities_compiled

.. autofunction:: usearch.capabilities_available

.. autoclass:: usearch.Key
    :members:

.. autoclass:: usearch.Status
    :members:
    :undoc-members:

.. autoclass:: usearch.Scalar
    :members:
    :undoc-members:

.. autoclass:: usearch.Residency
    :members:
    :undoc-members:

.. autoclass:: usearch.Traversal
    :members:
    :undoc-members:

.. autoclass:: usearch.Mood
    :members:
    :undoc-members:

========
Errors
========

.. autoexception:: usearch.Error
.. autoexception:: usearch.OutOfMemoryError
.. autoexception:: usearch.ShapeMismatchError
.. autoexception:: usearch.UnsupportedLayoutError
.. autoexception:: usearch.UnknownKindError
.. autoexception:: usearch.ReadOnlyError
.. autoexception:: usearch.InputOutputError
.. autoexception:: usearch.DeviceError
.. autoexception:: usearch.MalformedError
.. autoexception:: usearch.InternalError
.. autoexception:: usearch.CapacityOverflowError
.. autoexception:: usearch.UninitializedError
.. autoexception:: usearch.ConfigMismatchError
.. autoexception:: usearch.ContendedError
.. autoexception:: usearch.CancelledError
.. autoexception:: usearch.UnsupportedError
.. autoexception:: usearch.InvalidArgumentError

================
Other Modalities
================

.. autoclass:: usearch.SequenceIndex

.. autoclass:: usearch.ShapesIndex

.. autoclass:: usearch.GeospatialIndex

======
Keys
======

.. automodule:: usearch.keys
    :members:

======
IO
======

.. automodule:: usearch.io
    :members:

==========
Evaluation
==========

.. automodule:: usearch.eval
    :members:
