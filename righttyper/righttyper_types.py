from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import NewType, TypeVar, Self, TypeAlias
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
    args: "tuple[TypeInfo|str, ...]" = tuple()    # arguments within []

    code_id: CodeId = CodeId(0)     # if a callable, generator or coroutine, the CodeId
    is_bound: bool = False          # if a callable, whether bound
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


NoneTypeInfo = TypeInfo("", "None", type_obj=types.NoneType)    # FIXME make Singleton using __new__
AnyTypeInfo = TypeInfo("typing", "Any")


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    default: TypeInfo|None


@dataclass(eq=True, frozen=True)
class FuncInfo:
    func_id: FuncId
    args: tuple[ArgInfo, ...]



@dataclass
class Sample:
    args: tuple[TypeInfo, ...]
    yields: set[TypeInfo] = field(default_factory=set)
    sends: set[TypeInfo] = field(default_factory=set)
    returns: TypeInfo = NoneTypeInfo
    is_async: bool = False
    is_generator: bool = False
    self_type: TypeInfo | None = None


    def process(self) -> tuple[TypeInfo, ...]:
        retval = self.returns

        if self.is_generator:
            y = TypeInfo.from_set(self.yields)
            s = TypeInfo.from_set(self.sends)

            if self.is_async:
                retval = TypeInfo("typing", "AsyncGenerator", (y, s))
            else:
                retval = TypeInfo("typing", "Generator", (y, s, self.returns))
            
        type_data = (*self.args, retval)

        if self.self_type:
            class SelfTransformer(TypeInfo.Transformer):
                """Converts self_type to "typing.Self"."""

                def visit(vself, node: TypeInfo) -> TypeInfo:
                    if (
                        self.self_type and
                        node.type_obj and
                        self.self_type.type_obj in node.type_obj.__mro__
                    ):
                        return TypeInfo("typing", "Self")

                    if node == self.self_type:
                        return TypeInfo("typing", "Self")

                    return super().visit(node)


            tr = SelfTransformer()
            type_data = (*(tr.visit(arg) for arg in type_data),)

        return type_data
