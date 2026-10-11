"""Executes the `>>>` examples embedded in the C docstrings.

Every verb documents itself with a runnable example, colocated with its implementation as a
`char const doc_*[]`. This keeps them honest: if a signature changes, the example fails here.
`doctest.testmod` rather than pytest's `--doctest-modules`, which cannot collect from a C
extension.

Run:
    python -m pytest test/doctests.py -v
"""

from __future__ import annotations

import doctest
from array import array

import usearch

_OPTION_FLAGS: int = doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL
"""Whitespace and exception detail are layout, so neither fails an example."""

_DOCUMENTED: list[str] = [
    "Key",
    "Match",
    "Matches",
    "Error",
    "Executor",
    "Arena",
    "Deferred",
    "On",
    "DenseIndex",
    "DenseTraversalStats",
    "version",
    "capabilities_compiled",
    "capabilities_available",
]
"""The names carrying examples. Docstrings on a C extension's types are not reachable through
`testmod` on the package, so each type is searched directly."""


def streams_served() -> bool:
    """Whether this build serves search streams, which the engine declares before it serves them."""
    index = usearch.DenseIndex.init_ram(dimensions=2)
    index.reserve(1)
    try:
        index.search_stream(array("d", [0.1, 0.2])).close()
    except usearch.UnsupportedError:
        return False
    return True


def test_c_docstring_examples():
    """Every example embedded in the extension runs and produces what it claims, the search stream
    examples wherever the build serves streams."""
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(verbose=False, optionflags=_OPTION_FLAGS)
    attempted = 0
    streams = streams_served()
    for name in _DOCUMENTED:
        for test in finder.find(getattr(usearch, name), name, globs={"usearch": usearch}):
            if not test.examples or (not streams and "search_stream" in test.name):
                continue
            attempted += len(test.examples)
            runner.run(test)

    assert attempted > 0, "no doctests were discovered in the extension"
    assert runner.failures == 0, f"{runner.failures} of {attempted} embedded examples failed"


def test_pure_python_doctests():
    """The examples in the package's own modules, which are few by design."""
    from usearch import keys

    result = doctest.testmod(keys, optionflags=_OPTION_FLAGS, verbose=False)
    assert result.failed == 0
