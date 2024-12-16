from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType, TypeVar, Self, Iterator, Iterable

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

@dataclass(eq=True, frozen=True)
class FuncAnnotation:
    args: list[tuple[ArgumentName, Typename]]
    retval: Typename|None


Typename = NewType("Typename", str)


class TypenameSet:
    def __init__(self: Self, names: Iterable[Typename] = []) -> None:
        from collections import Counter

        self.items: Counter[Typename] = Counter()
        self.items.update(names)

    def __iter__(self: Self) -> Iterator[Typename]:
        return self.items.__iter__()

    def __contains__(self: Self, name: object) -> bool:
        return name in self.items

    def __len__(self: Self) -> int:
        return len(self.items)

    def update(self: Self, names: Iterable[Typename]) -> None:
        self.items.update(names)

    def frequency(self: Self, name: Typename) -> int:
        """Returns how often a type has been added to this set."""
        return self.items[name]


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    type_name_set: TypenameSet
