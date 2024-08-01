from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
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
