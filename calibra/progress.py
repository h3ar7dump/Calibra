"""
Lightweight terminal progress bar and tracking utilities.

Zero external dependencies — uses standard library only.
Automatically adapts to TTY (in-place dynamic refresh with `\\r`) and
non-TTY (clean single-line logs without terminal escape artifacts).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable, Iterator, Optional, TextIO


class ProgressBar:
    """
    Lightweight, zero-dependency terminal progress bar.

    Parameters
    ----------
    total       : total number of steps/items (None for indeterminate progress).
    description : prefix text describing the operation.
    stream      : output stream (defaults to sys.stderr to avoid polluting stdout).
    bar_width   : character width of the progress bar block.
    leave       : whether to keep the progress line after completion (default False).
    min_interval: minimum seconds between redraws in TTY mode to minimize overhead.
    """

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "",
        stream: Optional[TextIO] = None,
        bar_width: int = 25,
        leave: bool = False,
        min_interval: float = 0.05,
    ) -> None:
        self.total = total
        self.description = description
        self.stream = stream if stream is not None else sys.stderr
        self.bar_width = bar_width
        self.leave = leave
        self.min_interval = min_interval

        self.current = 0
        self.start_time = time.monotonic()
        self.last_update_time = 0.0
        self._is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._closed = False
        self._last_nontty_pct = -1

    def __enter__(self) -> ProgressBar:
        self.start_time = time.monotonic()
        self._render(initial=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def update(
        self,
        current: int,
        total: Optional[int] = None,
        message: str = "",
        force: bool = False,
    ) -> None:
        """Update progress state and render."""
        if self._closed:
            return
        self.current = current
        if total is not None:
            self.total = total

        now = time.monotonic()
        # Rate limit updates for TTY to save CPU unless finishing or forced
        is_final = self.total is not None and self.current >= self.total
        if force or is_final or (now - self.last_update_time >= self.min_interval):
            self.last_update_time = now
            self._render(message=message, is_final=is_final)

    def _render(self, message: str = "", initial: bool = False, is_final: bool = False) -> None:
        elapsed = max(0.001, time.monotonic() - self.start_time)
        desc = f"{self.description}: " if self.description else ""

        if self._is_tty:
            if self.total is not None and self.total > 0:
                fraction = min(1.0, max(0.0, self.current / self.total))
                pct = int(fraction * 100)
                filled = int(self.bar_width * fraction)
                bar = "█" * filled + "░" * (self.bar_width - filled)
                eta_str = ""
                if fraction > 0.05 and not is_final:
                    eta_sec = max(0, int((elapsed / fraction) - elapsed))
                    eta_str = f" [ETA {eta_sec}s]"
                msg_suffix = f" - {message}" if message else ""
                line = f"\r{desc}[{bar}] {pct}% ({self.current}/{self.total}){eta_str}{msg_suffix}"
            else:
                spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                spin = spinners[int(elapsed * 10) % len(spinners)]
                msg_suffix = f" {message}" if message else ""
                line = f"\r{desc}{spin} {self.current} processed ({elapsed:.1f}s){msg_suffix}"

            # Pad line to overwrite previous longer text
            self.stream.write(line.ljust(80)[:120])
            self.stream.flush()
        else:
            # Non-TTY mode (CI, pipes): log cleanly at milestones
            if initial:
                self.stream.write(f"{desc}starting...\n")
                self.stream.flush()
            elif self.total is not None and self.total > 0:
                pct = int((self.current / self.total) * 100)
                if is_final or (pct >= self._last_nontty_pct + 25):
                    self._last_nontty_pct = pct
                    msg_suffix = f" ({message})" if message else ""
                    self.stream.write(
                        f"{desc}{pct}% ({self.current}/{self.total}) [{elapsed:.1f}s]{msg_suffix}\n"
                    )
                    self.stream.flush()

    def close(self) -> None:
        """Finalize the progress bar."""
        if self._closed:
            return
        self._closed = True
        if self._is_tty:
            if self.leave:
                self.stream.write("\n")
            else:
                # Clear the line
                self.stream.write("\r" + " " * 80 + "\r")
            self.stream.flush()


def track(
    iterable: Iterable[Any],
    description: str = "",
    total: Optional[int] = None,
    stream: Optional[TextIO] = None,
) -> Iterator[Any]:
    """
    Wrap an iterable with a ProgressBar.

    Usage:
        for item in track(episodes, description="Decoding videos"):
            process(item)
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None

    with ProgressBar(total=total, description=description, stream=stream) as pbar:
        for idx, item in enumerate(iterable):
            yield item
            pbar.update(idx + 1)
