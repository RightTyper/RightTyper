import signal
import sys
from types import CodeType, FrameType
from typing import Any, Callable, Dict, Optional

from righttyper.righttyper_utils import (
    TOOL_ID,
    TOOL_NAME,
    get_sampling_interval,
)

_EVENTS = frozenset({
    sys.monitoring.events.PY_START,
    sys.monitoring.events.PY_RETURN,
    sys.monitoring.events.PY_YIELD,
    sys.monitoring.events.CALL,
})


def register_monitoring_callbacks(
    enter_function: Callable[[bool, CodeType], Any],
    call_handler: Callable[[CodeType, int, object, object], Any],
    exit_function: Callable[[CodeType, int, Any], object],
    yield_function: Callable[[CodeType, int, Any], object],
    ignore_annotations: bool,
) -> None:
    """Set up tracking for all enters, calls, exits, and yields."""
    event_set = 0
    for event in _EVENTS:
        event_set |= event

    sys.monitoring.set_events(TOOL_ID, event_set)

    fns : Dict[Any, Callable[..., Any]]  = {
        sys.monitoring.events.PY_START : (lambda x, y: enter_function(ignore_annotations, x)),
        sys.monitoring.events.CALL : call_handler,
        sys.monitoring.events.PY_RETURN : exit_function,
        sys.monitoring.events.PY_YIELD : yield_function,
    }

    for event in fns:
        sys.monitoring.register_callback(
            TOOL_ID,
            event,
            fns[event],
        )


def reset_monitoring() -> None:
    """Clear all monitoring of events."""
    for event in _EVENTS:
        sys.monitoring.register_callback(TOOL_ID, event, None)
    for id in range(3, 5):
        try:
            sys.monitoring.set_events(
                id,
                sys.monitoring.events.NO_EVENTS,
            )
        except ValueError:
            pass
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 0)


def setup_timer(
    func: Callable[[int, Optional[FrameType]], None],
) -> None:
    signal.signal(signal.SIGALRM, func)
    signal.setitimer(
        signal.ITIMER_REAL,
        0.01,
    )


def setup_tool_id() -> None:
    global TOOL_ID
    while True:
        try:
            sys.monitoring.use_tool_id(TOOL_ID, TOOL_NAME)
            break
        except Exception:
            TOOL_ID += 1
