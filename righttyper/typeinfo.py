import typing
from typing import Iterator, Final
import types
from dataclasses import dataclass, replace, field
from righttyper.righttyper_types import CodeId
from righttyper.righttyper_utils import normalize_module_name


# The typing module does not define a type for such "typing special forms".
type SpecialForms = typing.Any|typing.Never

# What is allowed in TypeInfo.args
type TypeInfoArg = TypeInfo|str|types.EllipsisType

@dataclass(eq=True, frozen=True)
class TypeInfo:
    module: str
    name: str
    args: tuple[TypeInfoArg, ...] = tuple()    # arguments within []

    # These fields are included for convenience, but don't affect what type is meant
    code_id: CodeId | None = field(default=None, compare=False)  # if a callable, generator or coroutine, the CodeId
    is_bound: bool = field(default=False, compare=False)    # if a callable, whether bound
    type_obj: type|SpecialForms|None = field(default=None, compare=False)
    is_unknown: bool = field(default=False, compare=False)  # for UnknownTypeInfo; indicates we don't know the type.
    typevar_index: int = field(default=0, compare=False)
    typevar_name: str|None = field(default=None, compare=False) # TODO delete me?

    # Indicates equivalence to typing.Self. Note that is_self may be true for one
    # type (class) in one trace, but false for the exact same type in another,
    # so that is_self matters for equivalence
    is_self: bool = field(default=False, compare=True)


    def __str__(self) -> str:
        if self.typevar_name: # FIXME subclass?
            return self.typevar_name

        # We can't use type_obj here because we need to clear them before using 'multiprocessing',
        # since type objects aren't pickleable
        if (self.module, self.name) == ('types', 'UnionType') and self.args: # FIXME subclass?
            return "|".join(str(a) for a in self.args)
        
        if self.args or self.name == '':
            def arg2str(a: "TypeInfo|str|ellipsis") -> str:
                if a is Ellipsis:
                    return '...'
                if isinstance(a, str):
                    return f'"{a}"'
                return str(a)
            
            return (
                f"{self.fullname()}[" +
                    ", ".join(arg2str(a) for a in self.args) +
                "]"
            )

        return self.fullname()


    @staticmethod
    def list(args: "list[TypeInfo|str|ellipsis]") -> "TypeInfo":
        """Builds a list argument, such as the first argument of a Callable"""
        return TypeInfo('', '', args=tuple(args))   # FIXME subclass?


    def is_list(self) -> bool:
        """Returns whether this TypeInfo is really a list of types, created by our 'list' factory method above."""
        return self.name == '' and self.module == ''


    @staticmethod
    def from_type(t: type|SpecialForms, module: str|None = None, **kwargs) -> "TypeInfo":
        if t is types.NoneType:
            return NoneTypeInfo

        return TypeInfo(
            name=getattr(t, "__qualname__"), # sidesteps mypy errors for special forms
            module=normalize_module_name(getattr(t, "__module__") if module is None else module),
            type_obj=t,
            **kwargs
        )


    @staticmethod
    def from_set(s: "set[TypeInfo]", **kwargs) -> "TypeInfo":
        if not s:
            return NoneTypeInfo

        def expand_unions(t: "TypeInfo") -> Iterator["TypeInfo"]:
            # Don't merge unions designated as typevars, or the typevar gets lost.
            if t.type_obj is types.UnionType and not t.typevar_index:
                for a in t.args:
                    if isinstance(a, TypeInfo):
                        yield from expand_unions(a)
            else:
                yield t

        s = {expanded for t in s for expanded in expand_unions(t)}

        # If "Any" is present, the union reduces to "Any"
        if t := next((t for t in s if t.type_obj is typing.Any), None):
            return t

        # Any others subsume "Never" and "NoReturn", so delete them
        not_never = {t for t in s if t.type_obj not in (typing.Never, typing.NoReturn)}
        if not_never:
            s = not_never

        if len(s) == 1:
            return next(iter(s))

        return TypeInfo.from_type(
            types.UnionType,
            # 'None' at the end is seen as more readable
            args=tuple(sorted(s, key = lambda x: (x == NoneTypeInfo, str(x)))),
            **kwargs
        )


    def is_union(self) -> bool:
        return (self.module, self.name) == ('types', 'UnionType')


    def to_set(self) -> set["TypeInfo"]:
        if self.is_union():
            return set(t for t in self.args if isinstance(t, TypeInfo))

        return {self}


    def replace(self, **kwargs) -> "TypeInfo":
        return replace(self, **kwargs)


    def is_typevar(self) -> bool:
        """Returns whether this TypeInfo is (or encloses) a typevar."""
        return bool(self.typevar_index) or any(
            a.is_typevar()
            for a in self.args
            if isinstance(a, TypeInfo)
        )


    def fullname(self) -> str:
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


    def walk(self) -> Iterator["TypeInfo"]:
        """Walks through all TypeInfo objects within this TypeInfo."""
        for arg in self.args:
            if isinstance(arg, TypeInfo):
                yield from arg.walk()
        yield self


NoneTypeInfo: Final = TypeInfo("", "None", type_obj=types.NoneType)
UnknownTypeInfo: Final = TypeInfo.from_type(typing.Any, is_unknown=True)
AnyTypeInfo: Final = TypeInfo.from_type(typing.Any)


type CallTrace = tuple[TypeInfo, ...]
