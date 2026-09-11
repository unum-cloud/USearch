"""Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices."""

from usearch import lib as _extension
from usearch.lib import *  # noqa: F401,F403

__version__ = _extension.__version__

del _extension
