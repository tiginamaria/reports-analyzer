"""The model lock lets an interactive caller in between a background job's batches."""

import threading
import time

from app.retrieval.embedder import ModelLock


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.001)


def test_interactive_caller_goes_before_the_next_background_batch():
    lock = ModelLock()
    order: list[str] = []
    chat_is_waiting = threading.Event()

    def background():
        for batch in range(3):
            with lock.held():
                order.append(f"batch{batch}")
                if batch == 0:
                    chat_is_waiting.wait(timeout=5)  # the chat arrives while batch 0 runs

    def interactive():
        with lock.held(interactive=True):
            order.append("chat")

    worker = threading.Thread(target=background)
    worker.start()
    wait_until(lock.locked)
    chat = threading.Thread(target=interactive)
    chat.start()
    wait_until(lambda: lock._interactive_waiting == 1)
    chat_is_waiting.set()
    worker.join(5)
    chat.join(5)
    assert order == ["batch0", "chat", "batch1", "batch2"]


def test_background_callers_take_turns_without_an_interactive_waiter():
    lock = ModelLock()
    order: list[str] = []

    def job(name):
        for _ in range(3):
            with lock.held():
                order.append(name)
                time.sleep(0.001)

    threads = [threading.Thread(target=job, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert sorted(order) == ["a", "a", "a", "b", "b", "b"]
    assert not lock.locked()
