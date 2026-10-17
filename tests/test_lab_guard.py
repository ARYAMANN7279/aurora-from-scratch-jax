"""Check the replacement lab guard without accessing or signalling a real GPU job."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def guard(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/lab/gpu_guard.py"
    spec = importlib.util.spec_from_file_location("gpu_guard", path)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    monkeypatch.setattr(g, "ROOT", tmp_path)
    monkeypatch.setattr(g.signal, "signal", lambda *args: None)
    monkeypatch.setattr(g.time, "sleep", lambda *args: None)
    return g


def test_guard_excludes_busy_hot_and_zero_cards_and_sets_child_environment(guard, monkeypatch):
    g = guard
    cards = [["0", "uuid0", "20", "Not Active", "Not Active"],
             ["1", "uuid1", "20", "Not Active", "Not Active"],
             ["2", "uuid2", "85", "Not Active", "Not Active"],
             ["3", "uuid3", "20", "Not Active", "Not Active"]]
    monkeypatch.setattr(g, "snapshot", lambda: (cards, [("uuid1", 99)]))
    calls = []

    class Child:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unwanted/library")
    monkeypatch.setenv("BASH_ENV", "/unwanted/startup")
    monkeypatch.setattr(g.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)) or Child())
    assert g.main(["python", "run.py"]) == 0
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "uuid3"
    assert calls[0][1]["env"]["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    assert "LD_LIBRARY_PATH" not in calls[0][1]["env"] and "BASH_ENV" not in calls[0][1]["env"]
    assert calls[0][1]["start_new_session"]


def test_guard_stops_only_its_misplaced_group(guard, monkeypatch):
    g = guard
    row = ["1", "uuid1", "20", "Not Active", "Not Active"]
    states = iter([([row], []), ([row], []), ([row], [("uuid0", 123), ("uuid0", 999)])])
    monkeypatch.setattr(g, "snapshot", lambda: next(states))
    monkeypatch.setattr(g, "in_group", lambda pid, pgid: pid == pgid)

    class Child:
        pid, returncode = 123, None

        def poll(self):
            return self.returncode

    child = Child()
    monkeypatch.setattr(g.subprocess, "Popen", lambda *a, **kw: child)
    stopped = []

    def stop(c):
        stopped.append(c.pid)
        c.returncode = -15

    monkeypatch.setattr(g, "stop", stop)
    assert g.main(["python", "run.py"]) == 98
    assert set(stopped) == {123}


def test_guard_thermal_state(guard):
    assert guard.throttled(["1", "uuid1", "60", "Active", "Not Active"])
    assert not guard.throttled(["1", "uuid1", "60", "Not Active", "Not Active"])
