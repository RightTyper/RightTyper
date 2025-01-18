import threading
import inspect
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
        sys.monitoring.events.CALL,
    }
)


def register_monitoring_callbacks(
    enter_function: Callable[[CodeType, int], Any],
    call_handler: Callable[[CodeType, int, object, object], Any],
    exit_function: Callable[[CodeType, int, Any], object],
    yield_function: Callable[[CodeType, int, Any], object],
) -> None:
    """Set up tracking for all enters, calls, exits, and yields."""
    event_set = 0
    for event in _EVENTS:
        event_set |= event

    sys.monitoring.set_events(TOOL_ID, event_set)

    fns: dict[Any, Callable[..., Any]] = {
        sys.monitoring.events.PY_START: enter_function,
        sys.monitoring.events.CALL: call_handler,
        sys.monitoring.events.PY_RETURN: exit_function,
        sys.monitoring.events.PY_YIELD: yield_function,
    }

    for event in fns:
        sys.monitoring.register_callback(
            TOOL_ID,
            event,
            fns[event],
        )


def reset_monitoring(timer: threading.Timer, stop_event) -> None:
    """Clear all monitoring of events."""
    for event in _EVENTS:
        sys.monitoring.register_callback(TOOL_ID, event, None)

    try:
        sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.NO_EVENTS)
    except ValueError:
        pass
    if timer is not None:
        stop_event.set()
        timer.join()
    
def setup_timer(
    func: Callable[[int, FrameType|None], None],
) -> tuple[threading.Timer, threading.Event]:
    def wrapper():
        while not stop_event.is_set():
            # Use 0 as a placeholder if SIGALRM is not available
            _signum = signal.SIGALRM if hasattr(signal, 'SIGALRM') else 0  
            frame = inspect.currentframe()
            func(_signum, frame)
            stop_event.wait(0.01)  # Wait for 0.01 seconds or until the stop_event is set
    
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=wrapper)
    timer_thread.start()
    return timer_thread, stop_event


def setup_tool_id() -> None:
    global TOOL_ID
    while True:
        try:
            sys.monitoring.use_tool_id(TOOL_ID, TOOL_NAME)
            break
        except Exception:
            TOOL_ID += 1
