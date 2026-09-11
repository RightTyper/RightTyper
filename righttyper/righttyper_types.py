import hashlib
import marshal
from dataclasses import dataclass
from typing import Any, NamedTuple, NewType, Protocol, TypeGuard
from types import CodeType

from righttyper.logger import logger


class CallableWithCode(Protocol):
    """A callable that has a __code__ attribute."""
    @property
    def __code__(self) -> CodeType: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def code_of(obj: object) -> CodeType | None:
    """Return obj's ``__code__``, but only if it is a real code object.

    An object may synthesize the attribute rather than raise (mock's _Call answers
    any name with a child _Call, unhashable and not code), or raise something
    getattr won't suppress.  Both reach the process-global CALL handler, so catch
    broadly and check the result's type -- unlike safe_issubclass, a swallowed
    failure here only means "not code we can annotate".  See #193.

    Note this must read dynamically: inspect.getattr_static answers __code__ with
    the descriptor rather than the code object.
    """
    try:
        code = getattr(obj, '__code__', None)
    except Exception:
        logger.debug(f"code_of: {type(obj).__name__} raised on __code__", exc_info=True)
        return None

    return code if isinstance(code, CodeType) else None


def has_code(obj: object) -> TypeGuard[CallableWithCode]:
    """TypeGuard that narrows to CallableWithCode.

    CallableWithCode declares __code__ as a CodeType, so hasattr alone made this
    guard lie.  Defer to code_of so there is a single probe to keep honest.
    """
    return code_of(obj) is not None


Filename = NewType("Filename", str)
ArgumentName = NewType("ArgumentName", str)
VariableName = NewType("VariableName", str)
FunctionName = NewType("FunctionName", str)


def _content_hash(code: CodeType) -> int:
    """Hash code-object content, excluding source-line metadata.

    Catches actual bytecode/closure-body changes; ignores ``co_firstlineno``,
    the line table, and ``co_filename`` so source-revision drift (same
    content, different line) and installation-path differences both produce
    the same hash. ``marshal`` deterministically encodes tuples of
    primitive types across processes (Python's built-in ``hash`` does not).
    Nested code objects in ``co_consts`` are recursively reduced to their
    own hashes.
    """
    def _fields(c: CodeType) -> tuple:
        return (
            c.co_name, c.co_qualname, c.co_code,
            tuple(_content_hash(k) if isinstance(k, CodeType) else k
                  for k in c.co_consts),
            c.co_names, c.co_varnames, c.co_freevars, c.co_cellvars,
            c.co_argcount, c.co_posonlyargcount, c.co_kwonlyargcount,
            c.co_flags,
        )
    digest = hashlib.sha256(marshal.dumps(_fields(code))).digest()
    return int.from_bytes(digest[:8], 'little', signed=True)


class FuncLoc(NamedTuple):
    """Identifies a function by its source location (file + qualname + line).

    Constructible without a runtime code object — used wherever a function
    needs to be named but its bytecode isn't available (e.g., the AST
    transformer mapping a ``def`` back to its recorded annotations).
    """
    file_name: Filename
    func_name: FunctionName
    first_code_line: int


@dataclass(eq=True, order=True, frozen=True)
class CodeId:
    """Identifies a function by source location *and* bytecode content.

    Stricter than ``FuncLoc``: two functions sharing a location but
    differing in body (e.g., two lambdas on one line, property
    getter/setter at adjacent lines) remain distinct CodeIds. Used as the
    record-time identity in ``Observations.func_info`` and as
    ``TypeInfo.code_id`` so distinct Callable types stay distinct.
    """
    file_name: Filename
    func_name: FunctionName
    first_code_line: int
    # Line-number-independent content hash. Same content at different
    # lines (drift) ⇒ same bytecode_hash but different CodeIds; detected
    # at merge time. Different content at any lines ⇒ different
    # bytecode_hash ⇒ different CodeIds.
    bytecode_hash: int


    @staticmethod
    def from_code(code: CodeType) -> "CodeId":
        return CodeId(
            Filename(code.co_filename),
            FunctionName(code.co_qualname),
            code.co_firstlineno,
            _content_hash(code),
        )

    def to_loc(self) -> FuncLoc:
        return FuncLoc(self.file_name, self.func_name, self.first_code_line)


def cast_not_None[T](x: T | None) -> T:
    """Small utility to just cast off None from x's type"""
    return x    # type: ignore[return-value]
