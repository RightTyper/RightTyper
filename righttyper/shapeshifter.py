import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import sys

captured_shapes = defaultdict(set)


def transform_input(
    inp: List[Any],
) -> List[Tuple[Optional[str], ...]]:
    tup = inp[0]  # We assume all inputs have the same tuple lengths
    numargs = len(tup)
    all_vals: List[List[List[Optional[int]]]] = [
        [[] for j in range(len(tup[i]))] for i in range(numargs)
    ]
    output_vals: List[List[Optional[str]]] = [
        [None for j in range(len(tup[i]))] for i in range(numargs)
    ]
    hash_count: Dict[int, int] = defaultdict(int)
    hashes = [[0 for j in range(len(tup[i]))] for i in range(numargs)]
    # First, identify all the values that every shape dimension could hold;
    # then hash them all.
    for i in range(numargs):
        for j in range(len(tup[i])):
            v = []
            for input_entry in inp:
                v.append(input_entry[i][j])
            if len(set(v)) == 1:
                # All the same
                output_vals[i][j] = f"{v[0]}"
            all_vals[i][j] = v
            # Hash the entire column of values
            column_hash = hash(tuple(v))
            hash_count[column_hash] += 1
            hashes[i][j] = column_hash
    # Now post-process:
    # First, assign symbolic variables for each possible hash
    symbolic_variable = {}
    varindex = 0
    for k in hash_count:
        if hash_count[k] > 1:
            symbolic_variable[k] = f"dim{varindex}"
            varindex += 1
    # Now, replace every unassigned output value with its corresponding symbolic variable
    for i in range(numargs):
        for j in range(len(tup[i])):
            if not output_vals[i][j]:
                output_vals[i][j] = symbolic_variable[hashes[i][j]]
    output_tuples = [tuple(i) for i in output_vals]
    return output_tuples


def capture_shapes(func):
    def wrapper(*args, **kwargs):
        shapes = [arg.shape for arg in args if isinstance(arg, np.ndarray)]
        captured_shapes[func].add(tuple(shapes))
        return func(*args, **kwargs)

    return wrapper


def convert_to_jaxtyping(
    argument_datatypes: List[str],
    output_tuples: List[Tuple[Optional[str], ...]],
) -> List[str]:
    result = []
    for index, arg_tuple in enumerate(output_tuples):
        args = list(arg_tuple)
        s = []
        for arg in args:
            s.append(f"{arg}")
            s_str = " ".join(s)
        declaration = f'{argument_datatypes[index]}[Array, "{s_str}"]'
        result.append(declaration)

    return result


def print_annotations() -> None:
    for func in captured_shapes:
        tups = transform_input(list(captured_shapes[func]))
        n = len(tups)
        annotations = convert_to_jaxtyping(["Float" for i in range(n)], tups)
        print(f"{func}: {annotations}")


def handle_start(code, instruction_offset):
    func = code.co_qualname
    class_name = get_class_name_from_stack()
    print("START:", class_name, code.co_qualname)
    frame = inspect.currentframe()
    # Get info from the caller
    args, varargs, varkw, the_values = inspect.getargvalues(frame.f_back)
    shapes = []
    for k in the_values:
        val = the_values[k]
        if isinstance(val, (pd.DataFrame, np.ndarray)):
            shapes.append(val.shape)
        elif isinstance(val, torch.Tensor):
            shapes.append(tuple(val.shape))
    captured_shapes[func].add(tuple(shapes))


def handle_return(code, instruction_offset, retval):
    print("RETURN:", code.co_qualname)
    func_name = code.co_qualname
    filename = code.co_filename
    shapes = []
    if isinstance(retval, (pd.DataFrame, np.ndarray)):
        shapes.append(retval.shape)
    elif isinstance(retval, torch.Tensor):
        shapes.append(tuple(retval.shape))
    print(shapes)
    # We need to fix the logic here to keep the arguments and return values in lockstep.
    # captured_retval_shapes[func_name].add(tuples(shapes))
    # get retval
    pass


import inspect
from functools import reduce

TOOL_ID = 5
EVENTS = frozenset(
    {
        sys.monitoring.events.PY_START,
        sys.monitoring.events.PY_RETURN,
    }
)
event_set = reduce(lambda a, b: a | b, EVENTS)


def get_class_name_from_stack(
    max_depth: int = 5,
) -> Optional[str]:
    # Initialize the current frame
    current_frame = inspect.currentframe()
    try:
        # Move up in the stack frame by frame
        depth = 0
        while current_frame and depth < max_depth:
            # Check if 'self' is in the local variables of the frame
            if "self" in current_frame.f_locals:
                instance = current_frame.f_locals["self"]
                return str(
                    instance.__class__.__name__
                )  # Return class name of the instance
            # Move one level up in the stack
            current_frame = current_frame.f_back
            depth += 1
    finally:
        # Delete reference to the current frame to avoid reference cycles
        del current_frame
    return None


def init_monitoring():
    # Initialize the tool
    sys.monitoring.use_tool_id(TOOL_ID, "thing")
    sys.monitoring.set_events(TOOL_ID, event_set)


def enable_monitoring(handle_start, handle_return):
    # Register the event handlers
    sys.monitoring.register_callback(
        TOOL_ID,
        sys.monitoring.events.PY_START,
        handle_start,
    )
    sys.monitoring.register_callback(
        TOOL_ID,
        sys.monitoring.events.PY_RETURN,
        handle_return,
    )

    # Enable the events globally for the tool
    sys.monitoring.set_events(
        TOOL_ID,
        sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
    )


def disable_monitoring():
    for event in EVENTS:
        sys.monitoring.register_callback(TOOL_ID, event, None)
    sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.NO_EVENTS)


# @capture_shapes
def example_function(a, b, c):
    return c


init_monitoring()
enable_monitoring(handle_start, handle_return)

# Example usage
a = np.random.rand(2, 10)
b = np.random.rand(10, 100)
c = np.random.rand(10, 2)
example_function(a, b, c)
example_function(a, b, c)

a = np.random.rand(2, 101)
b = np.random.rand(101, 100)
c = np.random.rand(101, 2)
# example_function(a, b, c)

a_pt = torch.tensor(a)
b_pt = torch.tensor(b)
c_pt = torch.tensor(c)
example_function(a_pt, b_pt, c_pt)

disable_monitoring()


print_annotations()
