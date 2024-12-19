from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType, TypeVar, Self, Iterator, Iterable, TypeAlias

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


# Valid non-None TypeInfo.type_obj types: allows static casting
# 'None' away in situations where mypy doesn't recognize it.
TYPE_OBJ_TYPES: TypeAlias = type

@dataclass(eq=True, frozen=True)
class TypeInfo:
    module: str
    name: str
    args: "tuple[TypeInfo|str, ...]" = tuple()    # arguments within [] in the Typename

    func: FuncInfo|None = None              # if a callable, the FuncInfo
    is_bound: bool = False                  # if a callable, whether bound
    type_obj: TYPE_OBJ_TYPES|None = None

    def __str__(self: Self) -> str:
        module = self.module + '.' if self.module else ''
        if self.args:
            return (
                f"{module}{self.name}[" +
                    ", ".join(str(a) for a in self.args) +
                "]"
            )

        return f"{module}{self.name}"

    @staticmethod
    def from_type(t: TYPE_OBJ_TYPES, **kwargs) -> "TypeInfo":
        return TypeInfo(t.__module__, t.__qualname__, type_obj=t, **kwargs)

    class Transformer:
        def visit(self, node: "TypeInfo") -> "TypeInfo":
            new_args = tuple(
                self.visit(arg) if isinstance(arg, TypeInfo) else arg
                for arg in node.args
            )
            if new_args != node.args:
                return TypeInfo(node.module, node.name, args=new_args,
                                func=node.func, is_bound=node.is_bound,
                                type_obj=node.type_obj)
            return node


TypeInfoSet: TypeAlias = set[TypeInfo]


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    type_set: TypeInfoSet
