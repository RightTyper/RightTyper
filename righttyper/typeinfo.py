import typing
from typing import Iterator, Final, Callable, Any
import types
from dataclasses import dataclass, replace, field
from righttyper.righttyper_types import CodeId
from righttyper.righttyper_utils import normalize_module_name


# The typing module does not define a type for such "typing special forms".
type SpecialForms = typing.Any|typing.Never

# What is allowed in TypeInfo.args
type TypeInfoArg = TypeInfo|str|types.EllipsisType|tuple[()]


@dataclass(eq=True, frozen=True)
class TypeInfo:
    """Holds information about a type."""

    module: str
    name: str
    args: tuple[TypeInfoArg, ...] = tuple()    # arguments within []

    # These fields are included for convenience, but don't affect what type is meant
    code_id: CodeId | None = field(default=None, compare=False) # if a callable, generator or coroutine, the CodeId
    is_bound: bool = field(default=False, compare=False)        # if a callable, whether bound
    type_obj: type|SpecialForms|None = field(default=None, compare=False)
    is_unknown: bool = field(default=False, compare=False)  # for UnknownTypeInfo; indicates we don't know the type.

    # Type pattern index (>0), when a pattern is detected.  Only UnionTypeInfo should have these.
    # Multiple can occur in a function call, in different order, so include in comparisons.
    typevar_index: int = field(default=0, compare=True)

    # Indicates equivalence to typing.Self. Note that is_self may be true for one
    # type (class) in one trace, but false for the exact same type in another,
    # so that is_self matters for equivalence
    is_self: bool = field(default=False, compare=True)


    @staticmethod
    def _arg2str(a: TypeInfoArg, modifier: Callable[["TypeInfo"], str|None]|None) -> str:
        if a is Ellipsis:
            return '...'
        if isinstance(a, str):
            return f'"{a}"'
        if isinstance(a, tuple):
            return str(a)
        return a.format(modifier)


    def format(self, modifier: Callable[["TypeInfo"], str|None]|None=None) -> str:
        """Formats this TypeInfo as a string.
           The optional 'modifier' is called for all TypeInfo, allowing it to render differently."""

        if modifier and (alternative := modifier(self)):
            return alternative

        if self.args:
            return (
                f"{self.fullname()}[" +
                    ", ".join(self._arg2str(a, modifier) for a in self.args) +
                "]"
            )

        return self.fullname()


    def __str__(self) -> str:
        return self.format()


    def __repr__(self) -> str:
        return f"({self.module}, {self.name}, {self.format()})"


    @staticmethod
    def list(args: list[TypeInfoArg]) -> "TypeInfo":
        """Builds a list, such as the first argument of a Callable"""
        return ListTypeInfo.from_type(ListTypeInfo, args=tuple(args))


    def is_list(self) -> bool:
        return False


    @classmethod
    def from_type(cls, t: type|SpecialForms, module: str|None = None, **kwargs: Any) -> "TypeInfo":
        if t is types.NoneType:
            return NoneTypeInfo

        return cls(
            name=getattr(t, "__qualname__"), # sidesteps mypy errors for special forms
            module=normalize_module_name(getattr(t, "__module__") if module is None else module),
            type_obj=t,
            **kwargs
        )


    @staticmethod
    def from_set(s: "set[TypeInfo]", empty_is_none=False, **kwargs: Any) -> "TypeInfo":
        if not s:
            return NoneTypeInfo if empty_is_none else TypeInfo.from_type(typing.Never)

        def expand_unions(t: "TypeInfo") -> Iterator["TypeInfo"]:
            # Don't merge unions designated as typevars, or the typevar gets lost.
            if t.is_union() and not t.typevar_index:
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

        # Remove "Never generics" (e.g., dict[Never, Never]) when a non-Never version
        # of the same container also exists (e.g., dict[str, str]).
        # We exclude immutable containers since an empty one cannot become non-empty.
        import collections.abc as abc
        never_generics = {
            t for t in s
            if t.args
            and t.type_obj not in (tuple, abc.Sequence, abc.Set, abc.Mapping)
            and isinstance(t.args[0], TypeInfo)
            and t.args[0].type_obj is typing.Never
        }
        if never_generics:
            # Keep Never generic only if no non-Never version of that container exists
            s -= never_generics
            s |= {
                t for t in never_generics
                if not any(t2.type_obj is t.type_obj for t2 in s)
            }

        if len(s) == 1:
            return next(iter(s))

        return UnionTypeInfo.from_type(
            UnionTypeInfo,
            # 'None' at the end is seen as more readable
            args=tuple(sorted(s, key = lambda x: (x == NoneTypeInfo, str(x)))),
            **kwargs
        )


    def is_union(self) -> bool:
        return False


    def to_set(self) -> set["TypeInfo"]:
        return {self}


    def replace(self, **kwargs: Any) -> "TypeInfo":
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


class UnionTypeInfo(TypeInfo):
    """Union (set[TypeInfo]) type, with its members as arguments."""


    def is_union(self) -> bool:
        return True


    def to_set(self) -> set["TypeInfo"]:
        return set(t for t in self.args if isinstance(t, TypeInfo))


    def format(self, modifier: Callable[["TypeInfo"], str|None]|None=None) -> str:
        if modifier and (alternative := modifier(self)):
            return alternative

        return "|".join(self._arg2str(a, modifier) for a in self.args)


class ListTypeInfo(TypeInfo):
    """Unnamed list type, typically used for Callable arguments."""


    def is_list(self) -> bool:
        return True


    def to_set(self) -> set["TypeInfo"]:
        return set(t for t in self.args if isinstance(t, TypeInfo))


    def format(self, modifier: Callable[["TypeInfo"], str|None]|None=None) -> str:
        return "[" + ", ".join(self._arg2str(a, modifier) for a in self.args) + "]"


NoneTypeInfo: Final = TypeInfo("", "None", type_obj=types.NoneType)
UnknownTypeInfo: Final = TypeInfo.from_type(typing.Any, is_unknown=True)
AnyTypeInfo: Final = TypeInfo.from_type(typing.Any)


type CallTrace = tuple[TypeInfo, ...]
