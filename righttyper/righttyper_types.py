from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
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
    args: list[tuple[ArgumentName, Typename|int]]

    # retval and yieldval are used to construct the return type
    # if we're not returning a generic. If we are, returns_generic
    # will be a non-None generic index
    retval: Typename
    yieldval: Typename|None = None

    returns_generic: int|None = None
    yields_generic: int|None = None

    # maps generic indices to typesets
    generics: dict[int, list[Typename]] = field(default_factory=lambda: defaultdict(list))

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


@dataclass
class Generic:
    arg_names: set[str]
    is_return: bool|None = None
    is_yield: bool|None = None
    index: int = -1

    def merge_generics(a: list[Self], b: list[Self]) -> list[Self]:
        # for every b that's a subset of someting in a:
        # break apart into subset and non-subset
        # if return agrees on both, keep it, otherwise don't
        # then we prune all single length generics, since
        # that's literally just an argument

        # we keep the generics we know are good in res
        res = []

        _a = a.copy()
        _b = b.copy()

        while len(_b):
            g1 = _b.pop(0)

            # we're going to iterate through all available at each point in time
            for g2 in _a:

                if not (g1.arg_names & g2.arg_names): continue
        
                # break apart into subset and leftover
                _a.remove(g2)

                # if we're not learning new information about the return, keep it as is
                # if we are, our subset will be the return if the part we're cutting from is also return
                # ex. {a,b}R, {a,b,c}R should result in {a,b}R -> res, {c} -> available
                subset_is_return = g2.is_return
                leftover_is_return = g2.is_return
                if g1.is_return is not None:
                    subset_is_return = g1.is_return and g2.is_return != False
                    leftover_is_return = g2.is_return != False and g1.is_return == False

                subset_is_yield = g2.is_yield
                leftover_is_yield = g2.is_yield
                if g1.is_yield is not None:
                    subset_is_yield = g1.is_yield and g2.is_yield != False
                    leftover_is_yield = g2.is_yield != False and g1.is_yield == False

                subset = Generic(g1.arg_names & g2.arg_names,
                    subset_is_return,
                    subset_is_yield)
                leftover_a = Generic(g2.arg_names - (g1.arg_names & g2.arg_names),
                    leftover_is_return,
                    leftover_is_yield)
                leftover_b = Generic(g1.arg_names - (g1.arg_names & g2.arg_names),
                    leftover_is_return,
                    leftover_is_yield)

                # add new generics back to a if they're big enough
                if subset.is_useful(): res.append(subset)
                if leftover_a.is_useful(): _a.append(leftover_a)
                if leftover_b.is_useful(): _b.append(leftover_b)

                break

        return res

    # a generic is useful if it spans across two arguemnts, or it
    # spans between >=1 argument and is either the return or yield
    def is_useful(self: Self):
        return len(self.arg_names) > 1 or len(self.arg_names) > 0 and self.is_return or self.is_yield

    def get(generics: list[Self], name: str) -> Self|None:
        for generic in generics:
            if name in generic.arg_names:
                return generic

        return None

