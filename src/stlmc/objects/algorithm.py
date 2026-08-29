import asyncio
import subprocess
import threading
import time
from abc import abstractmethod
from multiprocessing import *
from queue import Empty

from ..encoding.enumerate import *
from ..constraints.constraints import *
from ..objects.configuration import Configuration
from ..objects.goal import Goal
from ..objects.model import Model
from ..solver.abstract_solver import SMTSolver, ParallelSMTSolver
from ..util.logger import Logger
from ..util.print import Printer


class Algorithm:
    @abstractmethod
    def run(self, model: Model, goal: Goal, prop_dict: Dict, config: Configuration,
            solver: SMTSolver, logger: Logger, printer: Printer):
        pass

    @abstractmethod
    def set_debug(self, msg: str):
        pass


class AlgorithmRunner:
    @abstractmethod
    def run(self, solver: SMTSolver, const: Formula):
        pass

    @abstractmethod
    def check_sat(self):
        pass

    @abstractmethod
    def wait_and_check_sat(self):
        pass

    @abstractmethod
    def set_debug(self, msg: str):
        pass


async def solve(solver: SMTSolver, const: Formula):
    return await asyncio.wait_for(solver.solve(const), timeout=100000000.0)


def call_back(p):
    print(p)
    if p[0] == "False":
        print("not done!")
    else:
        print("done!")


class ParallelAlgRunner(AlgorithmRunner):
    def _check_sat(self):
        while True:
            try:
                result, model, smt_time = self.main_queue.get_nowait()
            except Empty:
                # no counterexample or unknown
                pass
            else:
                self.result = result
                if result == "False":
                    self.time += smt_time
                    self.model = model
                else:
                    self.model = None
                self.kill_all()
                break

    def check_sat(self):
        try:
            result, model, proc_id = self.main_queue.get_nowait()
        except Empty:
            # no counterexample or unknown
            return False, None
        else:
            self.increase_counter()
            # print(result)
            if result == "False":
                self.kill_all()
                return True, model
            else:
                if result == "Unknown":
                    self.saw_unknown = True
                self._drop(proc_id)
                return False, None

    def __init__(self, max_procs: int, query_budget=None):
        super().__init__()
        assert max_procs > 0
        print("max procs: {}".format(max_procs))
        self.procs: Set[subprocess.Popen] = set()
        self.sema = threading.Semaphore(max_procs)
        self.time = 0.0
        self.main_queue: Queue = Queue()

        self.result = None
        self.model = None
        self.debug_name = ""
        self.number = 0

        # Per-query wall-clock budget (None leaves each subprocess unbounded) and
        # the run-wide deadline (a monotonic instant, or None). A verdict that is
        # neither refuted nor a counterexample marks the run undecided.
        self.query_budget = query_budget
        self.deadline = None
        self.saw_unknown = False
        self.timed_out = False
        self._timers = dict()

    def _kill_proc(self, proc: subprocess.Popen):
        try:
            proc.terminate()
            proc.kill()
        except Exception:
            pass

    def _arm(self, proc: subprocess.Popen):
        budgets = []
        if self.query_budget is not None:
            budgets.append(float(self.query_budget))
        if self.deadline is not None:
            budgets.append(max(self.deadline - time.monotonic(), 0.0))
        if not budgets:
            return
        timer = threading.Timer(min(budgets), self._kill_proc, args=(proc,))
        timer.daemon = True
        timer.start()
        self._timers[id(proc)] = timer

    def _drop(self, proc_id: int):
        timer = self._timers.pop(proc_id, None)
        if timer is not None:
            timer.cancel()
        for proc in self.procs.copy():
            if id(proc) == proc_id:
                self.procs.discard(proc)


    def set_debug(self, msg: str):
        self.debug_name = msg

    def increase_counter(self):
        self.number += 1

    def run(self, solver: ParallelSMTSolver, const: Formula):
        assert isinstance(solver, ParallelSMTSolver)

        solver.set_file_name(self.debug_name)

        self.sema.acquire()
        proc = solver.process(self.main_queue, self.sema, const)
        self.procs.add(proc)
        self._arm(proc)


    def kill_all(self):
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        # A watchdog may already have killed a subprocess, so terminating or
        # reaping it again can raise; ignore that and drop it from the set.
        for proc in self.procs.copy():
            try:
                proc.terminate()
                proc.kill()
                proc.wait()
            except (ProcessLookupError, OSError):
                pass
            self.procs.discard(proc)

    def wait_and_check_sat(self):
        while len(self.procs) > 0:
            if self.deadline is not None:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    self.timed_out = True
                    self.kill_all()
                    break
                try:
                    result, model, proc_id = self.main_queue.get(timeout=remaining)
                except Empty:
                    self.timed_out = True
                    self.kill_all()
                    break
            else:
                result, model, proc_id = self.main_queue.get()
            self.increase_counter()
            if result == "False":
                self.kill_all()
                return True, model
            if result == "Unknown":
                self.saw_unknown = True
            self._drop(proc_id)
        return False, None


class NormalRunner(AlgorithmRunner):
    def check_sat(self):
        assert self.solver is not None and self.const is not None
        # Clamp the per-query budget to what remains of the run deadline; with no
        # time left the query is not started and the run is undecided.
        budget = self.query_budget
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self.timed_out = True
                self.solver = None
                self.const = None
                self.number = 0
                return False, None
            budget = remaining if budget is None else min(budget, remaining)
        if budget is not None and hasattr(self.solver, "set_query_budget"):
            self.solver.set_query_budget(budget)

        result, size = self.solver.solve(self.const)
        is_true = result == "False"
        if result == "Unknown":
            self.saw_unknown = True

        model = None
        if is_true:
            model = self.solver.make_assignment()
        self.time = self.solver.logger.get_duration_time("solving timer")
        self.solver = None
        self.const = None
        self.number = 0
        return is_true, model

    def __init__(self, query_budget=None):
        super().__init__()
        self.time = 0.0
        self.main_queue: Queue = Queue()
        self.solver = None
        self.const = None
        self.number = 0
        # Per-query budget handed to the dReal solver before each serial call
        # (None leaves it unbounded); an Unknown verdict marks the run undecided.
        self.query_budget = query_budget
        self.saw_unknown = False
        self.timed_out = False
        self.deadline = None

    def set_debug(self, msg: str):
        self.debug_name = msg

    def increase_counter(self):
        self.number += 1

    def run(self, solver: ParallelSMTSolver, const: Formula):
        assert isinstance(solver, ParallelSMTSolver)

        solver.set_file_name(self.debug_name)

        self.solver = solver
        self.const = const

    def kill_all(self):
        pass

    def wait_and_check_sat(self):
        return False, None