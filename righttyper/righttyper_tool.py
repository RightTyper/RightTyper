import sys
from types import CodeType
from typing import Any, Final
from collections.abc import Callable
import functools

from righttyper.righttyper_types import CallableWithCode, has_code
from righttyper.righttyper_utils import unwrap

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


def _call_handler(code: CodeType, offset: int, callable: Callable, arg0: object) -> Any:
    callee_code = getattr(callable, "__code__", None)

    # Check __dict__ directly to avoid triggering __getattr__ (e.g., on MagicMock)
    wrapped = (
        unwrap(callable)
        if "__wrapped__" in getattr(callable, "__dict__", ())
        else None
    )

    # Record wrapper->wrapped relationship for type propagation
    if (
        has_code(wrapped)
        and (wrapped_code := wrapped.__code__) in setup_code
    ):
        # For class instances, the executing code is __call__'s code
        wrapper_code = callee_code or getattr(
            getattr(type(callable), "__call__", None), "__code__", None
        )
        if wrapper_code and wrapper_code is not wrapped_code:
            wrapped_by[wrapper_code] = wrapped

    if callee_code in code_to_callable:
        call_mapping[(code, offset)] = callable
    elif (
        callee_code in setup_code
        or (
            wrapped is not None
            and (callee_code := getattr(callable := wrapped, "__code__", None)) in setup_code
        )
    ):
        call_mapping[(code, offset)] = callable
        code_to_callable[callee_code] = callable
    return sys.monitoring.DISABLE


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
    sys.monitoring.register_callback(TOOL_ID, events.CALL, _call_handler)

    # UNWIND and CALL are always global
    global_events = events.PY_UNWIND | events.CALL

    if USE_LOCAL_EVENTS:
        sys.monitoring.set_events(TOOL_ID, global_events)
    else:
        sys.monitoring.set_events(
            TOOL_ID,
            global_events|events.PY_START|events.PY_YIELD|events.PY_RETURN
        )


def shutdown_monitoring() -> None:
    sys.monitoring.register_callback(TOOL_ID, events.PY_START, None)
    sys.monitoring.register_callback(TOOL_ID, events.PY_YIELD, None)
    sys.monitoring.register_callback(TOOL_ID, events.PY_RETURN, None)
    sys.monitoring.register_callback(TOOL_ID, events.PY_UNWIND, None)
    sys.monitoring.register_callback(TOOL_ID, events.CALL, None)

    try:
        sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.NO_EVENTS)
    except ValueError:
        pass


setup_code: set[CodeType] = set()
enabled_code: set[CodeType] = set()

call_mapping: dict[tuple[CodeType, int], Callable] = {}
code_to_callable: dict[CodeType, Callable] = {}

# Maps wrapper code -> wrapped callable (must have __code__)
wrapped_by: dict[CodeType, CallableWithCode] = {}


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
    enabled_code.discard(code)
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
