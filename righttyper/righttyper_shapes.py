from collections import defaultdict
from typing import Any

from righttyper.righttyper_types import FuncInfo

# FIXME needs to be thread local
current_shape: dict[FuncInfo, list[tuple[int, ...]]] = defaultdict(list)

captured_shapes: dict[FuncInfo, set[tuple[int, ...]]] = defaultdict(set)


def transform_input(
    inp: list[Any],
) -> list[tuple[str|None, ...]]:
    tup = inp[0]  # We assume all inputs have the same tuple lengths
    numargs = len(tup)
    all_vals: list[list[list[int|None]]] = [
        [[] for j in range(len(tup[i]))] for i in range(numargs)
    ]
    output_vals: list[list[str|None]] = [
        [None for j in range(len(tup[i]))] for i in range(numargs)
    ]
    hash_count: dict[int, int] = defaultdict(int)
    hashes = [[0 for j in range(len(tup[i]))] for i in range(numargs)]
    # First, identify all the values that every shape dimension could hold;
    # then hash them all.
    for i in range(numargs):
        for j in range(len(tup[i])):
            v = []
            for input_entry in inp:
                try:
                    v.append(input_entry[i][j])
                except IndexError:
                    # FIXME - we should filter out the case where there are no shapes (all tuples are ()).
                    # print(f"{i=} {j=} {input_entry=}")
                    continue
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
    argument_datatypes: list[str],
    output_tuples: list[tuple[str|None, ...]],
) -> list[str]:
    result = []
    for index, arg_tuple in enumerate(output_tuples):
        args = list(arg_tuple)
        s = []
        s_str = ""
        for arg in args:
            s.append(f"{arg}")
            s_str = " ".join(s)
        if s_str:
            declaration = f'{argument_datatypes[index]}[{{}}, "{s_str}"]'
        else:
            declaration = "{}"
        result.append(declaration)

    return result


def update_arg_shapes(func: FuncInfo, the_values: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd
    import torch

    the_shapes: list[Any] = []
    for k in the_values:
        val = the_values[k]
        if isinstance(val, (pd.DataFrame, np.ndarray)):
            the_shapes.append(tuple(val.shape))
        elif isinstance(val, torch.Tensor):
            the_shapes.append(tuple(val.shape))
        else:
            the_shapes.append(tuple())
    current_shape[func].append(tuple(the_shapes))


def update_retval_shapes(func: FuncInfo, retval: Any) -> None:
    import numpy as np
    import pandas as pd
    import torch

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


def print_annotation(func: FuncInfo) -> list[str]:
    # No annotations if the shape was never captured
    if func not in captured_shapes:
        return []
    # No annotations if all captured shapes are empty tuples
    if all(
        all(s == () for s in shape) for shape in list(captured_shapes[func])
    ):
        return []
    tups = transform_input(list(captured_shapes[func]))
    n = len(tups)
    annotations = convert_to_jaxtyping(["Float" for i in range(n)], tups)
    return annotations


def print_annotations() -> None:
    for func in captured_shapes:
        print(f"{func}: {print_annotation(func)}")
