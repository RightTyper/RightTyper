from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import NewType, TypeVar, Self, TypeAlias, List, Iterator, cast
import collections.abc as abc
import types

T = TypeVar("T")

ArgumentName = NewType("ArgumentName", str)

Filename = NewType("Filename", str)
FunctionName = NewType("FunctionName", str)

CodeId = NewType("CodeId", int)     # obtained from id(code) where code is-a CodeType
FrameId = NewType("FrameId", int)   # similarly from id(frame)


@dataclass(eq=True, frozen=True)
class FuncId:
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
    args: "tuple[TypeInfo|str|ellipsis, ...]" = tuple()    # arguments within []

    # These fields are included for convenience, but don't affect what type is meant
    code_id: CodeId = field(default=CodeId(0), compare=False)   # if a callable, generator or coroutine, the CodeId
    is_bound: bool = field(default=False, compare=False)        # if a callable, whether bound
    type_obj: TYPE_OBJ_TYPES|None = field(default=None, compare=False)
    typevar_index: int = field(default=0, compare=False)
    typevar_name: str|None = field(default=None, compare=False) # TODO delete me?

    # Indicates equivalence to typing.Self. Note that is_self may be true for one
    # type (class) in one trace, but false for the exact same type in another,
    # so that is_self matters for equivalence
    is_self: bool = field(default=False, compare=True)


    def __str__(self: Self) -> str:
        if self.typevar_name: # FIXME subclass?
            return self.typevar_name

        # We can't use type_obj here because we need to clear them before using 'multiprocessing',
        # since type objects aren't pickleable
        if (self.module, self.name) == ('types', 'UnionType'): # FIXME subclass?
            return "|".join(str(a) for a in self.args)
        
        if self.args or self.name == '':
            def arg2str(a: TypeInfo|str|ellipsis) -> str:
                if a is Ellipsis:
                    return '...'
                if isinstance(a, str):
                    return f'"{a}"'
                return str(a)
            
            return (
                f"{self.qualname()}[" +
                    ", ".join(arg2str(a) for a in self.args) +
                "]"
            )

        return self.qualname()


    @staticmethod
    def list(args: "list[TypeInfo|str|ellipsis]") -> "TypeInfo":
        """Builds a list argument, such as the first argument of a Callable"""
        return TypeInfo('', '', args=tuple(args))   # FIXME subclass?


    @staticmethod
    def from_type(t: TYPE_OBJ_TYPES, module: str|None = None, **kwargs) -> "TypeInfo":
        if t == types.NoneType:
            return NoneTypeInfo

        return TypeInfo(
            name=t.__qualname__,
            module=(t.__module__ if module is None else module),
            type_obj=t,
            **kwargs
        )


    @staticmethod
    def from_set(s: "set[TypeInfo]", **kwargs) -> "TypeInfo":
        if not s:
            return NoneTypeInfo

        if len(s) == 1:
            return next(iter(s))

        def expand_unions(t: "TypeInfo") -> Iterator["TypeInfo"]:
            # don't merge unions designated as typevars, or the typevar gets lost.
            if t.type_obj is types.UnionType and not t.typevar_index:
                for a in t.args:
                    if isinstance(a, TypeInfo):
                        yield from expand_unions(a)
            else:
                yield t

        s = {ex for t in s for ex in expand_unions(t)}

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


    def qualname(self) -> str:
        return self.module + '.' + self.name if self.module else self.name


    class Transformer:
        def visit(self, node: "TypeInfo") -> "TypeInfo":
            new_args = tuple(
                self.visit(arg) if isinstance(arg, TypeInfo) else arg
                for arg in node.args
            )
            # Use identity rather than ==, as only non-essential attributes may have changed
            if any(old is not new for old, new in zip(node.args, new_args)):
                return node.replace(args=new_args)

            return node


NoneTypeInfo = TypeInfo("", "None", type_obj=types.NoneType)
UnknownTypeInfo = TypeInfo("typing", "Any")
AnyTypeInfo = TypeInfo("typing", "Any")


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    default: TypeInfo|None


@dataclass
class FunctionDescriptor:
    """Describes a function by name; stands in for a FunctionType where the function
       is a wrapper_descriptor (or possibly other objects), lacking __module__
    """
    __module__: str
    __qualname__: str


@dataclass(eq=True, frozen=True)
class FuncInfo:
    func_id: FuncId
    args: tuple[ArgInfo, ...]
    varargs: ArgumentName|None
    kwargs: ArgumentName|None
    overrides: types.FunctionType|FunctionDescriptor|None


CallTrace: TypeAlias = tuple[TypeInfo, ...]

@dataclass
class PendingCallTrace:
    args: tuple[TypeInfo, ...]
    yields: set[TypeInfo] = field(default_factory=set)
    sends: set[TypeInfo] = field(default_factory=set)
    returns: TypeInfo = NoneTypeInfo
    is_async: bool = False
    is_generator: bool = False
    self_type: TypeInfo | None = None
    self_replacement: TypeInfo | None = None


    def process(self) -> CallTrace:
        retval = self.returns

        if self.is_generator:
            y = TypeInfo.from_set(self.yields)
            s = TypeInfo.from_set(self.sends)

            if self.is_async:
                retval = TypeInfo.from_type(abc.AsyncGenerator, module="typing", args=(y, s))
            else:
                retval = TypeInfo.from_type(abc.Generator, module="typing", args=(y, s, self.returns))
            
        type_data = (*self.args, retval)

        if self.self_type:
            class SelfTransformer(TypeInfo.Transformer):
                """Replaces 'self' types with the type of the class that defines them,
                   also setting is_self for possible later replacement with typing.Self."""

                def visit(vself, node: TypeInfo) -> TypeInfo:
#                    if self.self_type: print(f"checking {str(node)} against {str(self.self_type)}")
                    if (
                        self.self_type
                        and self.self_replacement
                        and hasattr(node.type_obj, "__mro__")
                        and self.self_type.type_obj in cast(type, node.type_obj).__mro__
                    ):
#                        print(f"replacing {str(node)} with {str(self.self_replacement)}")
                        node = self.self_replacement.replace(is_self=True)

                    return super().visit(node)


            tr = SelfTransformer()
            type_data = (*(tr.visit(arg) for arg in type_data),)

        return type_data
