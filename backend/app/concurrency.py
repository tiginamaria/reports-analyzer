"""Bridges between the event loop and the worker threads that run blocking code.

PyMuPDF and the local models are synchronous, so the pipelines run them with
`asyncio.to_thread` (the LLM providers are async and need no thread). Some of that code
calls back into the pipeline (a progress callback after every page, a cancellation check
before every batch) and those callbacks need the database, which is async. `sync_callback`
turns an async callback into a blocking one the worker thread can call: it schedules the
coroutine on the loop and waits for the result, so an exception the callback raises
(`IngestionCancelled`) stops the worker exactly where the synchronous version did."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


def sync_callback[**P, R](
    callback: Callable[P, Coroutine[Any, Any, R]], loop: asyncio.AbstractEventLoop | None = None
) -> Callable[P, R]:
    """A blocking function for worker threads that runs `callback` on the event loop and
    returns (or raises) its result. Must be created on the loop, before the thread starts."""
    loop = loop or asyncio.get_running_loop()

    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        return asyncio.run_coroutine_threadsafe(callback(*args, **kwargs), loop).result()

    return call
