import sys
import traceback
from importlib.metadata import PackageNotFoundError, version

from ..driver.abstract_driver import *
from ..driver.base_driver import *
from ..exception.exception import *
from ..util.print import *


def _distribution_version() -> str:
    """Installed distribution version, or 'unknown' when it cannot be read."""
    try:
        return version("stlmc-x")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    """Run the model checker and return a process exit status.

    Returns the checker's exit status: 0 on a completed run and 1 on a failure.
    The run status is produced by the driver, which reports a failed run as 1; an
    exception that escapes the driver is also reported as 1. ``--version`` prints
    the distribution version and returns 0 without running the checker.
    """
    if "--version" in sys.argv[1:]:
        print(_distribution_version())
        return 0

    printer = ExceptionPrinter()
    try:
        driver_factory = BaseDriverFactory()

        stlmc = StlModelChecker()
        stlmc.create_env(driver_factory)
        status = stlmc.run()
    except NotSupportedError as E:
        printer.print_normal("system error: {}".format(E))
        return 1
    except OperationError as E:
        printer.print_normal("operation error: {}".format(E))
        return 1
    except ParsingError as E:
        printer.print_normal("parsing error: {}".format(E))
        return 1
    except Exception as E:
        printer.print_normal("error: {}".format(E))
        printer.print_normal(traceback.format_exc())
        return 1
    return status if isinstance(status, int) else 0