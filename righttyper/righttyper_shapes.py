from dataclasses import dataclass

import_failures = set()

try:
    import numpy as np
except ModuleNotFoundError:
    import_failures.add("numpy")

try:
    import pandas as pd
except ModuleNotFoundError:
    @dataclass
    class pd:
        DataFrame = type(list)
    import_failures.add("pandas")

try:
    import torch
except ModuleNotFoundError:
    @dataclass
    class torch:
        Tensor = type(list)
    import_failures.add("torch")

if len(import_failures) > 0:
    print(f"Warning: these missing imports limit RightTyper's shape analysis: {', '.join(list(import_failures))}")
    
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import sys
import inspect
from functools import reduce

# FIXME needs to be thread local
current_shape : Dict[str, List[Tuple[int, ...]]] = defaultdict(list)

captured_shapes : Dict[str, Set[Tuple[int, ...]]] = defaultdict(set)

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

def convert_to_jaxtyping(
    argument_datatypes: List[str],
    output_tuples: List[Tuple[Optional[str], ...]],
) -> List[str]:
    result = []
    for index, arg_tuple in enumerate(output_tuples):
        args = list(arg_tuple)
        s = []
        s_str = ""
        for arg in args:
            s.append(f"{arg}")
            s_str = " ".join(s)
        declaration = f'{argument_datatypes[index]}[Array, "{s_str}"]'
        result.append(declaration)

    return result

def update_arg_shapes(func, the_values):
    shapes = []
    for k in the_values:
        val = the_values[k]
        if isinstance(val, (pd.DataFrame, np.ndarray)):
            shapes.append(val.shape)
        elif isinstance(val, torch.Tensor):
            shapes.append(tuple(val.shape))
        else:
            shapes.append(tuple())
    current_shape[func].append(tuple(shapes))


def update_retval_shapes(func, retval):
    if len(current_shape[func]) == 0:
        return
    shapes = []
    if isinstance(retval, (pd.DataFrame, np.ndarray)):
        shapes.append(retval.shape)
    elif isinstance(retval, torch.Tensor):
        shapes.append(tuple(retval.shape))
    else:
        shapes.append(tuple())
    curr_shape = list(current_shape[func].pop())
    merged_shapes = tuple(curr_shape + shapes)
    captured_shapes[func].add(merged_shapes)
    # print(f"Shape for {func} = {merged_shapes}")
    
    
def print_annotations() -> None:
    for func in captured_shapes:
        tups = transform_input(list(captured_shapes[func]))
        n = len(tups)
        annotations = convert_to_jaxtyping(["Float" for i in range(n)], tups)
        print(f"{func}: {annotations}")

