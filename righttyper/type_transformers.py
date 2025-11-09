import typing
import types
import collections.abc as abc
from righttyper.typeinfo import TypeInfo, AnyTypeInfo, NoneTypeInfo
from righttyper.righttyper_utils import is_test_module
from righttyper.typemap import AdjustTypeNamesT
from righttyper.righttyper_runtime import get_type_name

import logging
from righttyper.logger import logger


class SelfT(TypeInfo.Transformer):
    """Renames types to typing.Self according to is_self."""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        if node.is_self:
            return TypeInfo("typing", "Self")

        return super().visit(node)


class NeverSayNeverT(TypeInfo.Transformer):
    """Removes uses of typing.Never, replacing them with typing.Any"""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        if node.type_obj is typing.Never:
            return AnyTypeInfo

        return super().visit(node)


class NoReturnToNeverT(TypeInfo.Transformer):
    """Converts typing.NoReturn to typing.Never,
       which is the more modern way to type a 'no return'"""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        if node.type_obj is typing.NoReturn:
            return TypeInfo.from_type(typing.Never) 

        return super().visit(node)


class ExcludeTestTypesT(TypeInfo.Transformer):
    """Removes test types."""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        if is_test_module(node.module):
            return AnyTypeInfo

        return super().visit(node)


class TypesUnionT(TypeInfo.Transformer):
    """Replaces types.UnionType with typing.Union and typing.Optional."""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)

        # Typevar nodes may be UnionType; there's no need to replace them, and
        # replacing them would prevent RightTyper from annotating as typevars.
        if node.type_obj is types.UnionType and not node.is_typevar():
            has_none = node.args[-1] == NoneTypeInfo
            non_none_count = len(node.args) - int(has_none)
            if non_none_count > 1:
                non_none = TypeInfo("typing", "Union", args=node.args[:non_none_count])
            else:
                assert isinstance(node.args[0], TypeInfo)
                non_none = node.args[0]

            if has_none:
                return TypeInfo("typing", "Optional", args=(non_none,))

            return non_none

        return node


class DepthLimitT(TypeInfo.Transformer):
    """Limits the depth of types (types within generic types)."""
    def __init__(self, limit: int):
        self._limit = limit
        self._level = -1
        self._maxLevel = -1

    def visit(self, node: TypeInfo) -> TypeInfo:
        # Don't count lists (such as the arguments in a Callable) as a level,
        # as it's not really a new type.
        if node.is_list():
            return super().visit(node)

        try:
            self._level += 1
            self._maxLevel = max(self._maxLevel, self._level)

            t = super().visit(node)

            if self._maxLevel > self._limit:
                # for containers, we can simply delete arguments (they default to Any)
                if (
                    (type(t.type_obj) is type and issubclass(t.type_obj, abc.Container))
                    or t.type_obj is abc.Callable
                ):
                    self._maxLevel = self._level
                    return t.replace(args=())

            return t
        finally:
            self._level -= 1


def _resolve_mock(ti: TypeInfo, adjuster: AdjustTypeNamesT|None) -> TypeInfo|None:
    """Attempts to map a test type, such as a mock, to a production one."""
    import unittest.mock as mock

    trace = [ti]
    t = ti
    while is_test_module(t.module):
        if not t.type_obj:
            return None

        non_unittest_bases = [
            b for b in t.type_obj.__bases__ if b not in (mock.Mock, mock.MagicMock)
        ]

        # To be conservative, we only recognize classes with a single base (besides any Mock).
        # If the only base is 'object', we couldn't find a non-mock module base.
        if len(non_unittest_bases) != 1 or (base := non_unittest_bases[0]) is object:
            return None

        t = get_type_name(base)
        if adjuster:
            t = adjuster.visit(t)
        trace.append(t)
        if len(trace) > 50:
            return None # break loops

    if t is ti:
        # 'ti' didn't need resolution.  Indicate no need to replace the type.
        return None

    if logger.level == logging.DEBUG:
        logger.debug(f"Resolved mock {' -> '.join([str(t) for t in trace])}")

    return t


class ResolveMocksT(TypeInfo.Transformer):
    """Resolves apparent test mock types to non-test ones."""

    def __init__(self, adjuster: AdjustTypeNamesT|None):
        self._adjuster = adjuster

    # TODO make mock resolution context sensitive, leaving test-only
    # objects unresolved within test code?
    def visit(self, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)
        if (resolved := _resolve_mock(node, self._adjuster)):
            return resolved
        return node


class GeneratorToIteratorT(TypeInfo.Transformer):
    """Converts Generator[X, None, None] -> Iterator[X]"""
    def visit(self, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)

        if (
            node.type_obj is abc.Generator
            and len(node.args) == 3
            and node.args[1] == NoneTypeInfo
            and node.args[2] == NoneTypeInfo
        ):
            return TypeInfo("typing", "Iterator", (node.args[0],))

        return node


class MakePickleableT(TypeInfo.Transformer):
    """Clears code and type_obj on all TypeInfo: annotations are pickled by 'multiprocessing',
       but these objects may not be pickleable.
    """
    def visit(self, node: TypeInfo) -> TypeInfo:
        if node.type_obj is not None:
            node = node.replace(type_obj=None)
        return super().visit(node)
