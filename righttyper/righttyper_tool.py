import sys
from types import CodeType
from typing import Any, Final
from collections.abc import Callable
import functools

TOOL_NAME: Final[str] = "righttyper"

def _setup_tool_id(tool_id: int) -> int:
    while 0 <= tool_id <= 5:    # only 0..5 supported by sys.monitoring
        try:
            sys.monitoring.use_tool_id(tool_id, TOOL_NAME)
            return tool_id

        except Exception:
            tool_id += 1

    raise RuntimeError("Unable to obtain a sys.monitoring tool id")

TOOL_ID: int = _setup_tool_id(3)
USE_LOCAL_EVENTS = True

events = sys.monitoring.events


def setup_monitoring(
    start_handler: Callable[[CodeType, int], Any],
    yield_handler: Callable[[CodeType, int, Any], object],
    return_handler: Callable[[CodeType, int, Any], object],
    unwind_handler: Callable[[CodeType, int, BaseException], Any],
) -> None:
    sys.monitoring.register_callback(TOOL_ID, events.PY_START, start_handler)
    sys.monitoring.register_callback(TOOL_ID, events.PY_YIELD, yield_handler)
    sys.monitoring.register_callback(TOOL_ID, events.PY_RETURN, return_handler)
    sys.monitoring.register_callback(TOOL_ID, events.PY_UNWIND, unwind_handler)

    if USE_LOCAL_EVENTS:
        # but UNWIND must always be global
        sys.monitoring.set_events(TOOL_ID, events.PY_UNWIND)
    else:
        sys.monitoring.set_events(
            TOOL_ID,
            events.PY_START|events.PY_YIELD|events.PY_RETURN|events.PY_UNWIND
        )


def shutdown_monitoring() -> None:
    sys.monitoring.register_callback(TOOL_ID, events.PY_START, None)
    sys.monitoring.register_callback(TOOL_ID, events.PY_YIELD, None)
    sys.monitoring.register_callback(TOOL_ID, events.PY_RETURN, None)
    sys.monitoring.register_callback(TOOL_ID, events.PY_UNWIND, None)

    try:
        sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.NO_EVENTS)
    except ValueError:
        pass


setup_code: set[CodeType] = set()
enabled_code: set[CodeType] = set()


def setup_monitoring_for_code(code: CodeType) -> None:
    """Enables sys.monitoring on the given code object."""
    if USE_LOCAL_EVENTS:
        sys.monitoring.set_local_events(TOOL_ID, code, events.PY_START|events.PY_YIELD|events.PY_RETURN)

    setup_code.add(code)
    enabled_code.add(code)

    for c in code.co_consts:
        if isinstance(c, CodeType):
            setup_monitoring_for_code(c)


def stop_events(code: CodeType) -> None:
    enabled_code.remove(code)
    if USE_LOCAL_EVENTS:
        sys.monitoring.set_local_events(TOOL_ID, code, events.NO_EVENTS)


def restart_events() -> None:
    if USE_LOCAL_EVENTS:
        disabled = setup_code - enabled_code
        enabled_code.update(setup_code)
        for code in disabled:
            sys.monitoring.set_local_events(TOOL_ID, code, events.PY_START|events.PY_YIELD|events.PY_RETURN)
    else:
        enabled_code.update(setup_code)
        sys.monitoring.restart_events()
