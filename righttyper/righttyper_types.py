from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from typing import NewType, TypeVar, Self, TypeAlias, Optional

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
    yieldval: Optional[Typename] = None

    returns_generic: Optional[int] = None
    yields_generic: Optional[int] = None

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
    is_return: Optional[bool] = None
    is_yield: Optional[bool] = None
    index: int = -1

    @staticmethod
    def merge_generics(a: list[Generic], b: list[Generic]) -> list[Generic]:
        """
        Takes two lists of `Generic` objects and combines them into an updated list.
        There are a couple complexities with this, but the main idea is as follows:

        Each generic in list a is a group of arguments that we *know* have the same
        type in all previous calls. Each generic in list b is a group of arguments
        that have the same calls as of the most recent function invocation. We need
        to square these two.

        We also must take into account the knowledge we have about whether or not any
        given generic also corresponds to either the return type or the yield type.
        Yields and returns are handled seperately, so we need to not change any known
        info about returns from a yield and vice versa. In this case, having either
        is_yield or is_return be `None` means "we don't know anything about this field"

        Args:
        a (list[Generic]): List of generics as known before this new call
        b (list[Generic]): List of potential generics we found from the most recent call

        Returns:
        list[Generic]: List of generics as known after this call
        """

        res = []

        for known in a:
            for new in b:
                if not (new.arg_names & known.arg_names):
                    continue

                # by default assume the same knowledge about yeild and return as original
                subset = Generic(new.arg_names & known.arg_names, known.is_return, known.is_yield)

                if new.is_return is not None:
                    # we use != False because we want it to still be true even if
                    # g2.is_return is None, since it being None means we know nothing
                    # about the return type
                    subset.is_return = new.is_return and known.is_return != False # noqa: E712

                if new.is_yield is not None:
                    subset.is_yield= new.is_yield and known.is_yield != False # noqa: E712

                if len(subset.arg_names) + bool(subset.is_return or subset.is_yield) > 1:
                    res.append(subset)

        return res

    @staticmethod
    def get(generics: list[Generic], name: str) -> Optional[Generic]:
        """
        Finds the generic an argument name corresponds to

        Args:
        generics (list[Generic]): list of generics to search through
        name (str): name of argument to get generic for

        Returns:
        Optional[Generic]: The found generic or None
        """
        return next(filter(lambda a: name in a.arg_names, generics), None)

