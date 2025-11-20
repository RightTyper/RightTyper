import typing
import types
import collections.abc as abc
from righttyper.typeinfo import TypeInfo, AnyTypeInfo, NoneTypeInfo
from righttyper.generalize import merged_types

import logging
from righttyper.logger import logger


class SelfT(TypeInfo.Transformer):
    """Renames types to typing.Self according to is_self."""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        if node.is_self:
            return TypeInfo.from_type(typing.Self)

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
    """Removes types from test modules."""

    def __init__(self, test_modules: set[str]) -> None:
        self._test_modules = test_modules

    def visit(self, node: TypeInfo) -> TypeInfo:
        if node.module in self._test_modules:
            return AnyTypeInfo

        return super().visit(node)


class TypesUnionT(TypeInfo.Transformer):
    """Replaces types.UnionType with typing.Union and typing.Optional."""
    def visit(vself, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)

        # Typevar nodes may be UnionType; there's no need to replace them, and
        # replacing them would prevent RightTyper from annotating as typevars.
        if node.is_union() and not node.is_typevar():
            has_none = node.args[-1] == NoneTypeInfo
            non_none_count = len(node.args) - int(has_none)
            if non_none_count > 1:
                non_none = TypeInfo.from_type(typing.Union, args=node.args[:non_none_count])
            else:
                assert isinstance(node.args[0], TypeInfo)
                non_none = node.args[0]

            if has_none:
                return TypeInfo.from_type(typing.Optional, args=(non_none,))

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
                    or t.type_obj in (abc.Callable, typing.Callable)
                ):
                    self._maxLevel = self._level
                    return t.replace(args=())

            return t
        finally:
            self._level -= 1


class GeneratorToIteratorT(TypeInfo.Transformer):
    """Converts Generator[X, None, None] -> Iterator[X]"""
    def visit(self, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)

        if (
            node.type_obj in (abc.Generator, typing.Generator)
            and len(node.args) == 3
            and type(arg0 := node.args[0]) is TypeInfo
            and node.args[1] == NoneTypeInfo
            and node.args[2] == NoneTypeInfo
        ):
            return TypeInfo.from_type(abc.Iterator, args=(
                arg0 if arg0.is_typevar() else merged_types(arg0.to_set()),
            ))
        elif (
            node.type_obj in (abc.AsyncGenerator, typing.AsyncGenerator)
            and len(node.args) == 2
            and type(arg0 := node.args[0]) is TypeInfo
            and node.args[1] == NoneTypeInfo
        ):
            return TypeInfo.from_type(abc.AsyncIterator, args=(
                arg0 if arg0.is_typevar() else merged_types(arg0.to_set()),
            ))

        return node


class MakePickleableT(TypeInfo.Transformer):
    """Clears type_obj on all TypeInfo, making them pickleable.
       Pickling is needed for saving, but also done by the multiprocessing module.
    """
    def visit(self, node: TypeInfo) -> TypeInfo:
        if node.type_obj is not None:
            node = node.replace(type_obj=None)
        return super().visit(node)


class LoadTypeObjT(TypeInfo.Transformer):
    """Loads TypeInfo's type_obj (e.g., cleared by MakePickleableT) by its module and name."""

    @classmethod
    def load_object(cls, node: TypeInfo) -> object:
        import importlib

        if node.is_list() or '.<locals>.' in node.name:
            return None

        if node.fullname() == 'None':
            return types.NoneType

        parts = node.name.split('.')
        modname = node.module if node.module else 'builtins'

        try:
            obj: object = importlib.import_module(modname)
        except:
            return None

        for part in parts:
            obj = getattr(obj, part, None)
        return obj
    

    def visit(self, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)
        if node.type_obj is None and (type_obj := self.load_object(node)):
            node = node.replace(type_obj=type_obj)
        return node
