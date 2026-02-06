from dataclasses import dataclass
from typing import Any, NewType, Protocol, TypeGuard
from types import CodeType


class CallableWithCode(Protocol):
    """A callable that has a __code__ attribute."""
    @property
    def __code__(self) -> CodeType: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def has_code(obj: object) -> TypeGuard[CallableWithCode]:
    """TypeGuard that narrows to CallableWithCode."""
    return hasattr(obj, '__code__')


Filename = NewType("Filename", str)
ArgumentName = NewType("ArgumentName", str)
VariableName = NewType("VariableName", str)
FunctionName = NewType("FunctionName", str)

@dataclass(eq=True, order=True, frozen=True)
class CodeId:
    file_name: Filename
    func_name: FunctionName
    first_code_line: int

    # if a <genexpr> or such, hash(code) to be able to differentiate same-line objects
    code_hash: int


    @staticmethod
    def from_code(code: CodeType) -> "CodeId":
        return CodeId(
            Filename(code.co_filename),
            FunctionName(code.co_qualname),
            code.co_firstlineno,
            hash(code) if code.co_qualname.endswith('>') else 0
        )


def cast_not_None[T](x: T | None) -> T:
    """Small utility to just cast off None from x's type"""
    return x    # type: ignore[return-value]
