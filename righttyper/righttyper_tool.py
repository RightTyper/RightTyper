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

_EVENTS = frozenset(
    {
        sys.monitoring.events.PY_START,
        sys.monitoring.events.PY_RETURN,
        sys.monitoring.events.PY_YIELD,
        sys.monitoring.events.PY_UNWIND,
    }
)

EVENT_BITSET = functools.reduce(int.__or__, _EVENTS, 0)

def register_monitoring_callbacks(
    start_handler: Callable[[CodeType, int], Any],
    return_handler: Callable[[CodeType, int, Any], object],
    yield_handler: Callable[[CodeType, int, Any], object],
    unwind_handler: Callable[[CodeType, int, BaseException], Any],
) -> None:
    """Set up tracking for all enters, exits, yields, and calls."""
    fns: dict[Any, Callable[..., Any]] = {
        sys.monitoring.events.PY_START: start_handler,
        sys.monitoring.events.PY_RETURN: return_handler,
        sys.monitoring.events.PY_YIELD: yield_handler,
        sys.monitoring.events.PY_UNWIND: unwind_handler,
    }

    for event in _EVENTS:
        sys.monitoring.register_callback(
            TOOL_ID,
            event,
            fns[event],
        )

    sys.monitoring.set_events(TOOL_ID, EVENT_BITSET)

def reset_monitoring() -> None:
    """Clear all monitoring of events."""
    for event in _EVENTS:
        sys.monitoring.register_callback(TOOL_ID, event, None)

    try:
        sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.NO_EVENTS)
    except ValueError:
        pass
