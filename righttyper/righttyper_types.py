from __future__ import annotations

import runstats
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    NewType,
    Set,
    Type,
    TypeVar,
)

T = TypeVar("T")

ArgumentName = NewType("ArgumentName", str)


class ArgumentType(Enum):
    positional = "PositionalArgument"
    vararg = "VariableArgument"
    kwarg = "KeywordArgument"


Filename = NewType("Filename", str)
FunctionName = NewType("FunctionName", str)


@dataclass(eq=True, frozen=True)
class FuncInfo:
    file_name: Filename
    func_name: FunctionName


Typename = NewType("Typename", str)


@dataclass
class TypenameFrequency:
    typename: Typename
    counter: int

    def __hash__(self: TypenameFrequency) -> int:
        return hash(self.typename)

    def __eq__(self: TypenameFrequency, other: Any) -> bool:
        if isinstance(other, TypenameFrequency):
            return self.typename == other.typename
        return False


TypenameSet = NewType("TypenameSet", Set[TypenameFrequency])


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    value_type: Type[Any]
    type_name_set: TypenameSet


@dataclass(eq=True, frozen=True)
class ImportDetails:
    object_name: str
    object_aliases: FrozenSet[str]
    import_module_name: str
    module_aliases: FrozenSet[str]


@dataclass(eq=True, frozen=True)
class ImportInfo:
    # 1. filename where the function lives
    function_fname: Filename
    # 2. filename where the class lives
    class_fname: Filename
    # 3. the name of the class
    class_name: str
    # 4. details for possible imports (see get_import_details).
    import_details: ImportDetails


# Track execution time of functions to adjust sampling
class ExecInfo(threading.local):
    def __init__(self) -> None:
        self.start_time: Dict[FuncInfo, List[float]] = defaultdict(list)
        self.execution_time: Dict[FuncInfo, runstats.Statistics] = defaultdict(
            runstats.Statistics
        )
        self.total_function_calls: int = 0
