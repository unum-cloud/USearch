"""One Proximity Graph for Search, Sparse Products & Attention, across RAM, VRAM & NVMe.

File: python/usearch/__init__.py
Author: Ash Vardanian
Date: May 10, 2023
"""

from usearch.lib import *  # noqa: F403
from usearch.lib import Status, __version__


async def _settled(call):
    """Answers what `call.wait()` answers, without parking a thread on the call.

    Polls `call.status` between sleeps that double from 1 µs to 1 ms, then calls `wait()`, which
    no longer blocks.
    """
    import asyncio

    pause = 1e-6
    while call.status == Status.DEFERRED:
        await asyncio.sleep(pause)
        pause = min(pause * 2, 1e-3)
    return call.wait()
