from dataclasses import dataclass, field
from typing import NewType, cast
import typing
import collections.abc as abc
import types
import inspect
from righttyper.typeinfo import TypeInfo, NoneTypeInfo


ArgumentName = NewType("ArgumentName", str)
VariableName = NewType("VariableName", str)

Filename = NewType("Filename", str)
FunctionName = NewType("FunctionName", str)

CodeId = NewType("CodeId", int)     # obtained from id(code) where code is-a CodeType
FrameId = NewType("FrameId", int)   # similarly from id(frame)


@dataclass(eq=True, order=True, frozen=True)
class FuncId:
    file_name: Filename
    first_code_line: int
    func_name: FunctionName


@dataclass(eq=True, frozen=True)
class FuncAnnotation:
    args: list[tuple[ArgumentName, TypeInfo]]   # TODO: make me a map?
    retval: TypeInfo
    varargs: str|None
    kwargs: str|None
    variables: list[tuple[VariableName, TypeInfo]]


@dataclass(eq=True, frozen=True)
class ModuleVars:
    variables: list[tuple[VariableName, TypeInfo]]


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


type CallTrace = tuple[TypeInfo, ...]

@dataclass
class PendingCallTrace:
    arg_info: inspect.ArgInfo
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

        if self.self_type and self.self_replacement:
            self_type = cast(TypeInfo, self.self_type)
            self_replacement = cast(TypeInfo, self.self_replacement)

            class SelfTransformer(TypeInfo.Transformer):
                """Replaces 'self' types with the type of the class that defines them,
                   also setting is_self for possible later replacement with typing.Self."""

                def visit(vself, node: TypeInfo) -> TypeInfo:
                    if (
                        hasattr(node.type_obj, "__mro__")
                        and self_type.type_obj in cast(type, node.type_obj).__mro__
                    ):
                        node = self_replacement.replace(is_self=True)

                    return super().visit(node)


            tr = SelfTransformer()
            type_data = (*(tr.visit(arg) for arg in type_data),)

        return type_data
