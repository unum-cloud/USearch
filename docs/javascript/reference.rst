JavaScript API Reference
========================

The JavaScript SDK is written in TypeScript, and the reference lives in the sources as
documentation comments, which the published ``.d.ts`` declarations carry into every editor.
``javascript/usearch.ts`` is the package entry point and re-exports two modules: ``types.ts``
with the shared vocabulary, and ``dense_index.ts`` with the dense modality.

=============
Dense Indexes
=============

``DenseIndex``
    A dense vector index, running every verb on the calling thread.
    Built by the static ``initRam``, ``initVram``, ``loadRam``, ``loadVram``, ``loadRamBuffer``,
    ``loadVramBuffer``, ``viewRam`` and ``viewNvme``, never by ``new``.
    ``on(executor)`` and ``deferred(executor)`` reach the same verbs on an executor, and ``free``
    releases the handle now rather than at collection.

``DenseIndexOn``
    The verbs of one index blocking on one executor, as ``index.on(executor)``: ``reserve``,
    ``clear``, ``describe``, ``health``, ``count``, ``contains``, ``insert``, ``refine``,
    ``search``, ``maxsim``, ``get``, ``multiply``, ``absorb``, ``save``, ``saveBuffer``, plus the
    ``cursor`` and ``searchStream`` walks.

``DenseIndexDeferredOn``
    The same verbs submitted to one executor and answered as ``Deferred`` calls, as
    ``index.deferred(executor)``.
    Cursors, streams and ``contains`` stay blocking, so this receiver offers none of them.

``DenseCursor``, ``DenseSearchStream``
    Resumable walks over every entry and over one query's neighborhood, each landing its batches
    in an arena.

``DenseMetric``, ``DenseTraversalStats``, ``DenseVectors``, ``DenseProducts``
    The metric vocabulary, per-call traversal telemetry, and the arena views ``get`` and
    ``multiply`` answer.

==================
Executors & Arenas
==================

``Executor``
    The workers a fanned-out verb runs on, shared across indexes:
    ``Executor.forRam``, ``Executor.forVram`` and ``Executor.forNvme`` spawn one, and ``drain``
    waits out every call submitted to it.

``Arena``
    The memory every output whose length scales with a call lands in.
    Each call rewinds the arena it is handed, so a steady loop allocates once, and a view of the
    previous call's outputs refuses its reads with ``Status.Uninitialized`` from then on.

``Matches``, ``MatchesRow``, ``Monolith``
    Views into an arena: the ranked runs ``search`` and ``maxsim`` write, one query's row of them,
    and the serialized checkpoint ``saveBuffer`` lands.

``Deferred``
    A submitted call: ``status`` polls without blocking, ``wait`` blocks and answers, and
    ``await`` runs that wait on the libuv pool.

==========
Vocabulary
==========

``Status``, ``Scalar``, ``Residency``, ``Traversal``, ``Mood``
    The closed sets every binding shares with the C interface.

``RowsView``
    A borrowed run of rows of one width, mirroring ``usearch_rows_view_t``; a single row may also
    travel as a bare TypedArray.

``keyFromInteger``, ``integerFromKey``, ``formatKey``, ``parseKey``, ``keyAt``
    Conversions between 128-bit keys, integers and their dashed UUID spelling.
