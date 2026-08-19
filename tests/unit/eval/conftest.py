from __future__ import annotations

from collections.abc import Iterator

import pytest

from strands_env.eval import LocalReporter


@pytest.fixture(autouse=True)
def close_local_reporters(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Close the file handle of every `LocalReporter` built during a test.

    `publish()` closes it in the real lifecycle; these tests assert on the file
    instead of running that far, and `Evaluator` builds its reporter internally so
    there is nothing for the test to close. Left open, GC raises ResourceWarning —
    which `filterwarnings = ["error"]` turns into a failure pinned on whichever test
    the collector happens to interrupt.
    """
    built: list[LocalReporter] = []
    original_init = LocalReporter.__init__

    def recording_init(self: LocalReporter, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]
        built.append(self)

    monkeypatch.setattr(LocalReporter, "__init__", recording_init)
    yield
    for reporter in built:
        if reporter._fh is not None:
            reporter._fh.close()
            reporter._fh = None
