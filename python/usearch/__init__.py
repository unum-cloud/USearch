"""One Proximity Graph for Search, Sparse Products & Attention across RAM, VRAM & NVMe.

File: python/usearch/__init__.py
Author: Ash Vardanian
Date: May 10, 2023
"""

from usearch.lib import *  # noqa: F403
from usearch.lib import Status, __version__


async def _settled(call):
    """Answers what `call.wait()` answers, without parking a thread on the call.

    Polls `call.status`, and while it is deferred, re-polls on the next turn of the running loop
    through a future `loop.call_soon` resolves, then calls `wait()`, which no longer blocks.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    while call.status == Status.DEFERRED:
        turn = loop.create_future()
        loop.call_soon(turn.set_result, None)
        await turn
    return call.wait()
