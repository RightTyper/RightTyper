import signal
import sys
from types import CodeType, FrameType
from typing import Any
from collections.abc import Callable

from righttyper.righttyper_utils import TOOL_ID, TOOL_NAME

_EVENTS = frozenset(
    {
        sys.monitoring.events.PY_START,
        sys.monitoring.events.PY_RETURN,
        sys.monitoring.events.PY_YIELD,
#        sys.monitoring.events.CALL,
    }
)


def register_monitoring_callbacks(
    enter_function: Callable[[CodeType, int], Any],
    exit_function: Callable[[CodeType, int, Any], object],
    yield_function: Callable[[CodeType, int, Any], object],
    call_handler: Callable[[CodeType, int, object, object], Any],
) -> None:
    """Set up tracking for all enters, exits, yields, and calls."""
    event_set = 0
    for event in _EVENTS:
        event_set |= event

    sys.monitoring.set_events(TOOL_ID, event_set)

    fns: dict[Any, Callable[..., Any]] = {
        sys.monitoring.events.PY_START: enter_function,
        sys.monitoring.events.PY_RETURN: exit_function,
        sys.monitoring.events.PY_YIELD: yield_function,
        sys.monitoring.events.CALL: call_handler,
    }

    for event in _EVENTS:
        sys.monitoring.register_callback(
            TOOL_ID,
            event,
            fns[event],
        )


def reset_monitoring() -> None:
    """Clear all monitoring of events."""
    for event in _EVENTS:
        sys.monitoring.register_callback(TOOL_ID, event, None)

    try:
        sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.NO_EVENTS)
    except ValueError:
        pass


def setup_tool_id() -> None:
    global TOOL_ID
    while True:
        try:
            sys.monitoring.use_tool_id(TOOL_ID, TOOL_NAME)
            break
        except Exception:
            TOOL_ID += 1
