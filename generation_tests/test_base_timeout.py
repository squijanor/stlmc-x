"""Regression tests for the base checker's dReal time budget.

Covers the config surface (``[common] solver-timeout``), the per-query and
total wall-clock budgets in the parallel and serial runners, and the rule that
an undecided backend query marks the run undecided rather than a refutation.

The tests are self-contained: no dReal or yices binary and no model or
benchmark file. The serial and parallel timeouts run against a stub subprocess
that sleeps, and the propagation is checked on the runner flag that
``EnumerateAlgorithm.run`` turns into ``Unknown``.
"""
import os
import subprocess
import threading
import time

import pytest

from stlmc.constraints.constraints import BoolVal
from stlmc.encoding.enumerate import resolve_solver_timeout
from stlmc.objects.algorithm import NormalRunner, ParallelAlgRunner
from stlmc.parser.config_visitor import ConfigVisitor
from stlmc.solver.abstract_solver import ParallelSMTSolver
from stlmc.solver.dreal import dRealSolver
from stlmc.util.logger import Logger

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CFG = os.path.join(_REPO, "src", "stlmc", "default.cfg")


# --------------------------------------------------------------------------- #
#  duck-typed config and stub solvers/processes
# --------------------------------------------------------------------------- #
class _Sec:
    def __init__(self, values):
        self._values = values

    def is_argument_in(self, key):
        return key in self._values

    def get_value(self, key):
        return self._values[key]


class _Cfg:
    def __init__(self, sections):
        self._sections = sections

    def get_section(self, name):
        return self._sections[name]


class _DummyProc:
    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self):
        pass


class _HangParallelSolver(ParallelSMTSolver):
    """process() spawns a real sleeping subprocess and, like dRealSolver,
    reports a killed process as Unknown."""

    def __init__(self):
        self.file_name = ""

    def set_file_name(self, name):
        self.file_name = name

    def process(self, main_queue, sema, const):
        proc = subprocess.Popen(["sleep", "1000"])

        def watch():
            proc.communicate()  # returns once the runner kills it
            result = "Unknown" if proc.returncode != 0 else "True"
            main_queue.put((result, None, id(proc)))
            sema.release()

        threading.Thread(target=watch, daemon=True).start()
        return proc


_HangParallelSolver.__abstractmethods__ = frozenset()


class _FakeSerial(ParallelSMTSolver):
    """Records the budget it is handed and returns a fixed verdict."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.budget = "unset"
        self.logger = Logger()

    def set_file_name(self, name):
        pass

    def set_query_budget(self, seconds):
        self.budget = seconds

    def solve(self, const=None, *args, **kwargs):
        self.logger.reset_timer()
        self.logger.start_timer("solving timer")
        self.logger.stop_timer("solving timer")
        return self.verdict, 0

    def make_assignment(self):
        return None


_FakeSerial.__abstractmethods__ = frozenset()


def _hang_script(tmp_path):
    path = tmp_path / "hang.sh"
    path.write_text("#!/bin/sh\nexec sleep 1000\n")
    path.chmod(0o755)
    return str(path)


def _drain(verdicts):
    """Feed verdicts through the real ParallelAlgRunner drain and return it."""
    runner = ParallelAlgRunner(4)
    for verdict in verdicts:
        proc = _DummyProc()
        runner.procs.add(proc)
        runner.main_queue.put((verdict, None, id(proc)))
    found, model = runner.wait_and_check_sat()
    return found, model, runner


# --------------------------------------------------------------------------- #
#  1. config acceptance and parsing of solver-timeout
# --------------------------------------------------------------------------- #
def test_config_with_solver_timeout_is_accepted(tmp_path):
    cv = ConfigVisitor()
    base = cv.parse_from_file(_DEFAULT_CFG)
    cfg_file = tmp_path / "m.cfg"
    cfg_file.write_text(
        'common {\n bound = 2\n time-bound = 2\n solver = "dreal"\n'
        ' solver-timeout = 30\n}\n'
        'dreal {\n ode-order = 5\n ode-step = 0.001\n}\n')
    cfg = cv.parse_from_file(str(cfg_file), base)
    assert cfg.get_section("common").get_value("solver-timeout") == "30"


@pytest.mark.parametrize("raw, expected", [
    (None, None), ("", None), ("off", None), ("none", None), ("0", None),
    ("30", 30.0), ("12.5", 12.5),
])
def test_resolve_solver_timeout_values(raw, expected):
    values = {} if raw is None else {"solver-timeout": raw}
    assert resolve_solver_timeout(_Cfg({"common": _Sec(values)})) == expected


@pytest.mark.parametrize("raw", ["-5", "abc", "inf", "nan"])
def test_resolve_solver_timeout_rejects_bad(raw):
    with pytest.raises(ValueError):
        resolve_solver_timeout(_Cfg({"common": _Sec({"solver-timeout": raw})}))


# --------------------------------------------------------------------------- #
#  2. serial dReal call over its budget returns Unknown
# --------------------------------------------------------------------------- #
def test_serial_dreal_query_timeout_returns_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solver = dRealSolver()
    solver.config = _Cfg({
        "dreal": _Sec({"executable-path": _hang_script(tmp_path)}),
        "common": _Sec({"time-horizon": "2", "time-bound": "2"}),
    })
    solver.append_logger(Logger())
    # skip declare-building; the point under test is the budget and the kill
    monkeypatch.setattr(solver, "get_declared_variables",
                        lambda const, th, tb: ([], 0))
    solver.set_query_budget(0.5)

    started = time.monotonic()
    result, _size = solver.solve(BoolVal("True"))
    assert result == "Unknown"
    assert time.monotonic() - started < 5.0


# --------------------------------------------------------------------------- #
#  3. parallel: a killed subprocess is reported Unknown
# --------------------------------------------------------------------------- #
def test_parallel_watchdog_kills_subprocess_and_marks_unknown():
    runner = ParallelAlgRunner(4, query_budget=0.5)
    runner.set_debug("t")
    try:
        runner.run(_HangParallelSolver(), None)
        started = time.monotonic()
        found, model = runner.wait_and_check_sat()
        assert time.monotonic() - started < 5.0
        assert found is False and model is None
        assert runner.saw_unknown is True
    finally:
        runner.kill_all()


# --------------------------------------------------------------------------- #
#  4. an Unknown candidate prevents a final True
#     run() returns "Unknown" iff the runner saw an undecided verdict, so the
#     flag it reads is the propagation under test.
# --------------------------------------------------------------------------- #
def test_unknown_candidate_marks_the_run_undecided():
    found, model, runner = _drain(["True", "Unknown", "True"])
    assert found is False and model is None
    assert runner.saw_unknown is True  # -> run() returns Unknown, not True


def test_only_refutations_leave_the_run_decidable():
    found, model, runner = _drain(["True", "True"])
    assert found is False and model is None
    assert runner.saw_unknown is False  # -> run() returns True


def test_a_counterexample_still_wins_over_an_unknown():
    runner = ParallelAlgRunner(4)
    ce, unknown = _DummyProc(), _DummyProc()
    runner.procs.update({ce, unknown})
    runner.main_queue.put(("Unknown", None, id(unknown)))
    runner.main_queue.put(("False", "MODEL", id(ce)))
    found, model = runner.wait_and_check_sat()
    assert found is True and model == "MODEL"


# --------------------------------------------------------------------------- #
#  5. total timeout clamps an in-flight serial call
# --------------------------------------------------------------------------- #
def test_serial_query_clamped_to_remaining_deadline():
    runner = NormalRunner(query_budget=300.0)
    runner.deadline = time.monotonic() + 2.0
    runner.set_debug("t")
    solver = _FakeSerial("True")
    runner.run(solver, "const")
    runner.check_sat()
    assert isinstance(solver.budget, float)
    assert solver.budget <= 2.0 and solver.budget < 300.0


def test_serial_no_time_left_is_undecided_without_a_query():
    runner = NormalRunner(query_budget=300.0)
    runner.deadline = time.monotonic() - 1.0
    runner.set_debug("t")
    solver = _FakeSerial("False")
    runner.run(solver, "const")
    found, model = runner.check_sat()
    assert found is False and model is None
    assert runner.timed_out is True
    assert solver.budget == "unset"  # no time left, so no query was issued