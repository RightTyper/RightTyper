import collections.abc as abc
import typing
import types
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import logging
from pathlib import Path
from righttyper.righttyper_types import TypeInfo, UnknownTypeInfo
from righttyper.righttyper_runtime import is_test_module
from righttyper.logger import logger
from righttyper.righttyper_utils import source_to_module_fqn


class TypeMap:
    """Maps type objects to a canonical name for the type, if possible."""

    @dataclass(eq=True)
    class TypeName:
        module_parts: list[str] = field(default_factory=list)
        name_parts: list[str] = field(default_factory=list)
        is_private: bool = False

        def to_strings(self) -> tuple[str, str]:
            return ".".join(self.module_parts), ".".join(self.name_parts)


    def __init__(self, main_globals: dict[str, typing.Any]|None):
        self._map: dict[type, tuple[str, str]] = self.build_map(main_globals)


    def find(self, t: type) -> tuple[str, str]|None:
        """Given a type object, return its module and qualified name as strings."""
        return self._map.get(t, None)


    def build_map(self, main_globals: dict[str, typing.Any]|None) -> dict[type, tuple[str, str]]:
        work_map: dict[type, list[TypeMap.TypeName]] = defaultdict(list)
        known_modules: set[str] = set()

        for m in list(sys.modules): # list() in case it changes while we're working
            if m != '__main__':     # this would be RightTyper's main
                self._add_types_from(
                    work_map,
                    sys.modules[m].__dict__,
                    m.split('.'), [],
                    is_private=(m.startswith("_") or "._" in m)
                )

        if main_globals:
            if not (
               (main_spec := main_globals.get('__spec__'))
               and main_spec.origin
               and (main_name := source_to_module_fqn(Path(main_spec.origin)))
            ):
                main_name = "__main__"

            self._add_types_from(
                work_map,
                main_globals,
                main_name.split('.'),
                [],
                is_private=False
            )

        def get_name(t: type) -> str|None:
            return getattr(t, "__qualname__", getattr(t, "__name__"))

        def typename_key(t: type, tn: TypeMap.TypeName) -> tuple[int, ...]:
            module, name = tn.to_strings()
            # str() because __module__ might be a getset_attribute (hello, cython)
            t_package = str(t.__module__).split('.')[0]
            return (
                # prefer non-test to test
                is_test_module(module),
                # prefer public to private
                tn.is_private,
                # prefer top-level X if defined in top-level X or _X
                not (t_package == tn.module_parts[0] or t_package == "_" + tn.module_parts[0]),
                # prefer shorter to longer (in components)
                len(tn.module_parts)+len(tn.name_parts),
                # prefer where's defined
                name != get_name(t),
            )

        search_map: dict[type, tuple[str, str]] = dict()

        for t in work_map:
            typename_list = sorted(work_map[t], key=lambda tn: typename_key(t, tn))
            mod_and_name = typename_list[0].to_strings()
            if mod_and_name[0] == 'builtins':
                mod_and_name = ('', mod_and_name[1])

            search_map[t] = mod_and_name

            if logger.level == logging.DEBUG:
                for tn in typename_list:
                    logger.debug(f"TypeMap {t.__module__}.{get_name(t)} {'.'.join(tn.to_strings())}")

            if False:
                for tn in typename_list:
                    #if typename_list[0].to_strings() != (t.__module__, get_name(t)):
                    print(f"TypeMap {t.__module__}.{get_name(t)} " +
                              f"{'.'.join(tn.to_strings())} {typename_key(t, tn)}")
        return search_map


    def _add_types_from(
        self,
        work_map: dict[type, list[TypeName]],
        dunder_dict: abc.Mapping[str, typing.Any],
        mod_parts: list[str],
        name_parts: list[str],
        *,
        is_private: bool,
        objs_in_path: list[object]|None = None
    ) -> None:
        """Recursively explores a module (or type), updating 'work_map'.
           'work_map': the map beint built
           'dunder_dict': the dictionary to space
           'mod_parts': the module name, split on '.'
           'name_parts': the qualified name parts, split on '.'
           'objs_in_path': set of objects being recursed on, for loop avoidance
        """

        if objs_in_path is None:
            objs_in_path = []

        dunder_all = (
            set(da)
            if isinstance((da := dunder_dict.get('__all__')), (list, tuple))
            else None
        )

        for name, obj in list(dunder_dict.items()):
            name_is_private = (
                is_private
                or (dunder_all is not None and name not in dunder_all)
                or name.startswith("_")
            )

            if (
                isinstance(obj, (type, types.ModuleType))
                # also include typing's special definitions; must be hashable to use as dict key
                or (
                    dunder_dict is typing.__dict__
                    and isinstance(obj, abc.Hashable)
                    and hasattr(obj, "__name__")
                )
            ):
                # Some module objects are really namespaces, like "sys.monitoring"; they
                # don't show up in sys.modules. We want to process any such, but leave others
                # to be processed on their own from sys.modules
                if isinstance(obj, types.ModuleType) and obj.__name__ in sys.modules:
                    continue

                new_name_parts = name_parts + [name]

                if not isinstance(obj, types.ModuleType):
                    work_map[typing.cast(type, obj)].append(
                        self.TypeName(
                            mod_parts,
                            new_name_parts,
                            is_private=name_is_private
                        )
                    )

                if isinstance(obj, (type, types.ModuleType)) and obj not in objs_in_path:
                    self._add_types_from(
                        work_map,
                        obj.__dict__,
                        mod_parts,
                        new_name_parts,
                        is_private=name_is_private,
                        objs_in_path=objs_in_path + [obj]
                    )


class AdjustTypeNamesT(TypeInfo.Transformer):
    """Adjust types' module and name by looking their type_obj on TypeMap."""
    def __init__(vself, main_globals: dict[str, typing.Any] | None):
        vself.type_map = TypeMap(main_globals)

    def visit(vself, node: TypeInfo) -> TypeInfo:
        if (
            node.type_obj
            and node.type_obj not in (
                types.NoneType,
                # FIXME temporary: righttyper_runtime generates these as "typing.X"
                abc.Callable, abc.Generator, abc.AsyncGenerator, abc.Coroutine
            )
        ):
            if (mod_and_name := vself.type_map.find(node.type_obj)):
                if mod_and_name != (node.module, node.name):
                    node = node.replace(module=mod_and_name[0], name=mod_and_name[1])
            else:
                # TODO how to check local names?
                if '.<locals>.' not in node.name:
                    return UnknownTypeInfo

        return super().visit(node)

