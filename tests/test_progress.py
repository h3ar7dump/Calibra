"""Tests for progress bar and progress tracking utilities (calibra/progress.py)."""

from __future__ import annotations

import io

from calibra.progress import ProgressBar, track


class DummyTTYStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class DummyNonTTYStream(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_progress_bar_tty():
    stream = DummyTTYStream()
    with ProgressBar(total=10, description="Processing", stream=stream, bar_width=20) as pbar:
        for i in range(1, 11):
            pbar.update(i, message=f"item {i}")

    output = stream.getvalue()
    assert "Processing" in output
    assert "100%" in output or "10/10" in output
    assert "\r" in output


def test_progress_bar_non_tty():
    stream = DummyNonTTYStream()
    with ProgressBar(total=4, description="NonTTY Task", stream=stream) as pbar:
        for i in range(1, 5):
            pbar.update(i, message=f"step {i}")

    output = stream.getvalue()
    assert "NonTTY Task" in output
    assert "\n" in output


def test_progress_bar_indeterminate():
    stream = DummyTTYStream()
    with ProgressBar(total=None, description="Loading", stream=stream, min_interval=0.0) as pbar:
        pbar.update(1, message="working...")
        pbar.update(2, message="done")

    output = stream.getvalue()
    assert "Loading" in output
    assert "done" in output


def test_track_iterator():
    stream = DummyTTYStream()
    items = [1, 2, 3, 4, 5]
    processed = []
    for x in track(items, description="Iterating", stream=stream):
        processed.append(x * 2)

    assert processed == [2, 4, 6, 8, 10]
    output = stream.getvalue()
    assert "Iterating" in output
