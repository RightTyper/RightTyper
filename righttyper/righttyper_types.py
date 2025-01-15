from __future__ import annotations

from dataclasses import dataclass, replace, field
from enum import Enum
from typing import NewType, TypeVar, Self, TypeAlias
import types

T = TypeVar("T")

ArgumentName = NewType("ArgumentName", str)

Filename = NewType("Filename", str)
FunctionName = NewType("FunctionName", str)

Typename = NewType("Typename", str)


@dataclass(eq=True, frozen=True)
class FuncInfo:
    file_name: Filename
    first_code_line: int
    func_name: FunctionName


@dataclass(eq=True, frozen=True)
class FuncAnnotation:
    args: list[tuple[ArgumentName, TypeInfo]]   # TODO: make me a map?
    retval: TypeInfo


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
    typevar_index: int = 0
    typevar_name: str|None = None   # TODO delete me?


    def __str__(self: Self) -> str:
        if self.typevar_name: # FIXME subclass?
            return self.typevar_name

        if self.module == "types" and self.name == "UnionType": # FIXME subclass?
            return "|".join(str(a) for a in self.args)
        
        module = self.module + '.' if self.module else ''
        if self.args:
            # TODO: fix callable arguments being strings
            # if self.module == "typing" and self.name == "Callable":
            #     return f"{module}{self.name}[[" + \
            #         ", ".join(str(a) for a in self.args[:-1]) + \
            #         f"], {str(self.args[-1])}]"
            
            return (
                f"{module}{self.name}[" +
                    ", ".join(str(a) for a in self.args) +
                "]"
            )

        return f"{module}{self.name}"


    @staticmethod
    def from_type(t: TYPE_OBJ_TYPES, module: str|None = None, **kwargs) -> "TypeInfo":
        if t == types.NoneType: return NoneTypeInfo

        return TypeInfo(
            name=t.__qualname__,
            module=(t.__module__ if module is None else module),
            type_obj=t,
            **kwargs
        )


    @staticmethod
    def from_set(s: "TypeInfoSet", **kwargs) -> "TypeInfo":
        if not s:
            return NoneTypeInfo

        if len(s) == 1:
            return next(iter(s))

        return TypeInfo(
            module='types',
            name='UnionType',
            type_obj=types.UnionType,
            # 'None' at the end is seen as more readable
            args=tuple(sorted(s, key = lambda x: (x == NoneTypeInfo, str(x)))),
            **kwargs
        )


    def replace(self, **kwargs) -> "TypeInfo":
        return replace(self, **kwargs)


    def is_typevar(self) -> bool:
        """Returns whether this TypeInfo is (or encloses) a typevar."""
        return bool(self.typevar_index) or any(
            a.is_typevar()
            for a in self.args
            if isinstance(a, TypeInfo)
        )


    class Transformer:
        def visit(self, node: "TypeInfo") -> "TypeInfo":
            new_args = tuple(
                self.visit(arg) if isinstance(arg, TypeInfo) else arg
                for arg in node.args
            )
            if new_args != node.args:
                return node.replace(args=new_args)

            return node


# FIXME make Singleton using __new__
NoneTypeInfo = TypeInfo("", "None", type_obj=types.NoneType)


TypeInfoSet: TypeAlias = set[TypeInfo]


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    default: TypeInfo|None


@dataclass
class Sample:
    args: tuple[TypeInfo, ...]
    yields: TypeInfoSet = field(default_factory=TypeInfoSet)
    returns: TypeInfo = NoneTypeInfo


    def process(self) -> tuple[TypeInfo, ...]:
        retval = self.returns
        if len(self.yields):
            y = TypeInfo.from_set(self.yields)
            is_async = False

            # FIXME capture send type and switch to Generator/AsyncGenerator if any sent

            if len(self.yields) == 1:
                y = next(iter(self.yields))
                if str(y) == "builtins.async_generator_wrapped_value":
                    y = TypeInfo("typing", "Any")  # FIXME how to unwrap the value without waiting on it?
                    is_async = True

            if self.returns is NoneTypeInfo:
                # Note that we are unable to differentiate between an implicit "None"
                # return and an explicit "return None".
                # FIXME return value doesn't matter for AsyncIterator
                iter_type = "AsyncIterator" if is_async else "Iterator"
                retval = TypeInfo("typing", iter_type, (y,))

            else:
                retval = TypeInfo("typing", "Generator", (y, TypeInfo("typing", "Any"), self.returns))

        return (*self.args, retval)
