import inspect
import builtins
from dataclasses import dataclass, field
from types import CodeType, FrameType, FunctionType, GeneratorType
from collections import defaultdict
import collections.abc as abc
from pathlib import Path
import logging
from righttyper.logger import logger
from righttyper.righttyper_types import ArgumentName, VariableName, Filename, CodeId, cast_not_None
from righttyper.typeinfo import TypeInfo, NoneTypeInfo, CallTrace
from typing import Final, Any, NewType, overload, cast
import typing
from righttyper.observations import Observations, FuncInfo, OverriddenFunction, ArgInfo
from righttyper.variable_capture import code2variables
from righttyper.options import run_options
from righttyper.righttyper_utils import (
    source_to_module_fqn, get_main_module_fqn, skip_this_file,
    detected_test_files, detected_test_modules, is_test_module,
    normalize_module_name
)
from righttyper.type_id import find_function, unwrap, get_value_type, get_type_name, hint2type, PostponedArg0
from righttyper.typemap import TypeMap, AdjustTypeNamesT, CheckTypeNamesT


# Singleton used to differentiate from None
NO_OBJECT: Final = object()


FrameId = NewType("FrameId", int)   # obtained from id(frame) where code is-a FrameType


# Overloads so we don't have to always write FrameId(id(code)), etc.
@overload
def id(obj: FrameType) -> FrameId: ...
@overload
def id(obj: object) -> int: ...
def id(obj: FrameType|object) -> FrameId|int:
    return builtins.id(obj)


class PendingCallTrace:
    """Builds a call trace."""

    def __init__(
        self,
        arg_info: inspect.ArgInfo,
        co_flags: int,
        self_type: TypeInfo|None,
        self_replacement: TypeInfo|None
    ) -> None:
        self.arg_info = arg_info
        self.args_start = self._get_arg_types(arg_info)
        self.yields: set[TypeInfo] = set()
        self.sends: set[TypeInfo] = set()
        self.is_async = bool(co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE))
        self.is_generator=bool(co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR))
        self.self_type = self_type
        self.self_replacement = self_replacement


    @staticmethod
    def _get_arg_types(arg_info: inspect.ArgInfo) -> tuple[TypeInfo, ...]:
        """Computes the types of the given arguments."""
        return (
            *(
                get_value_type(arg_info.locals[arg_name]) if arg_name in arg_info.locals else None
                for arg_name in arg_info.args
            ),
            *(
                (
                    TypeInfo.from_set({
                        get_value_type(val) for val in arg_info.locals[arg_info.varargs]
                    }, empty_is_none=True)
                    if arg_info.varargs in arg_info.locals else None,
                )
                if arg_info.varargs else ()
            ),
            *(
                (
                    TypeInfo.from_set({
                        get_value_type(val) for val in arg_info.locals[arg_info.keywords].values()
                    }, empty_is_none=True)
                    if arg_info.keywords in arg_info.locals else None,
                )
                if arg_info.keywords else ()
            )
        )


    def finish(self, retval: TypeInfo) -> CallTrace:
        if self.is_generator:
            y = TypeInfo.from_set(self.yields, empty_is_none=True)
            s = TypeInfo.from_set(self.sends, empty_is_none=True)

            if self.is_async:
                retval = TypeInfo.from_type(abc.AsyncGenerator, args=(y, s))
            else:
                retval = TypeInfo.from_type(abc.Generator, args=(y, s, retval))
            
        # Arguments may change value (and, in particular, empty containers may be added to)
        # during the function execution, so we sample a 2nd time at the end.
        args_now = self._get_arg_types(self.arg_info)
        type_data: tuple[TypeInfo, ...] = (
            *tuple(
                at_start if now is None else TypeInfo.from_set({at_start, now})
                for at_start, now in zip(self.args_start, args_now)
            ),
            retval
        )

        if self.self_type and self.self_replacement:
            self_type = self.self_type
            self_replacement = self.self_replacement

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


class ObservationsRecorder:
    def __init__(self) -> None:
        # Finds FuncInfo by their CodeType
        self._code2func_info: dict[CodeType, FuncInfo] = {}

        # Started, but not (yet) completed traces
        self._pending_traces: dict[CodeType, dict[FrameId, PendingCallTrace]] = defaultdict(dict)

        # Object attributes: class_key -> attr_name -> set[TypeInfo]
        self._object_attributes: dict[object, dict[VariableName, set[TypeInfo]]] = defaultdict(lambda: defaultdict(set))

        # Class attributes: class_key -> attr_name -> set[TypeInfo]
        self._class_attributes: dict[object, dict[VariableName, set[TypeInfo]]] = defaultdict(lambda: defaultdict(set))

        self._obs = Observations()


    def needs_more_traces(self, code: CodeType) -> bool:
        if (func_info := self._code2func_info.get(code)):
            traces = func_info.traces

            # Require a minimum number of traces to help stabilize the estimate
            if (n := traces.total()) < run_options.trace_min_samples:
                return True

            if n >= run_options.trace_max_samples:
                return False

            if (sum(c == 1 for c in traces.values()) / n) <= run_options.trace_type_threshold:
                return False

#            logger.info(f"{code=} {traces}")

        return True


    def record_module(
        self,
        code: CodeType,
        frame: FrameType
    ) -> None:
        # TODO handle cases where modules are loaded more than once, e.g. through pytest
        if code.co_filename and code.co_filename not in self._obs.source_to_module_name:
            if (modname := frame.f_globals.get('__name__', None)):
                if modname == "__main__":
                    modname = get_main_module_fqn()
            else:
                modname = source_to_module_fqn(Path(code.co_filename))

            assert modname
            self._obs.source_to_module_name[Filename(code.co_filename)] = modname


    def record_function(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo,
        overrides: OverriddenFunction|None
    ) -> None:
        """Records that a function was visited."""

        if code not in self._code2func_info:
            arg_names = (
                *(a for a in arg_info.args),
                *((arg_info.varargs,) if arg_info.varargs else ()),
                *((arg_info.keywords,) if arg_info.keywords else ())
            )

            defaults = get_defaults(code, frame)

            self._code2func_info[code] = func_info = FuncInfo(
                CodeId.from_code(code),
                tuple(
                    ArgInfo(ArgumentName(name), defaults.get(name))
                    for name in arg_names
                ),
                ArgumentName(arg_info.varargs) if arg_info.varargs else None,
                ArgumentName(arg_info.keywords) if arg_info.keywords else None,
                overrides
            )
            self._obs.func_info[func_info.code_id] = func_info


    def record_start(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo
    ) -> None:
        """Records a function start."""

        # print(f"record_start {code.co_qualname} {arg_types}")

        self.record_module(code, frame)

        # CO_NEWLOCALS, when set within a CodeType's co_flags, indicates that
        # a new "locals" dictionary is created; it roughly indicates a function
        # call, as the new dictionary is created for the new scope.
        if (code.co_flags & inspect.CO_NEWLOCALS):
            self_type, self_replacement, overrides = get_self_type(code, arg_info)
            self.record_function(code, frame, arg_info, overrides)

            self._pending_traces[code][id(frame)] = PendingCallTrace(
                arg_info, code.co_flags, self_type, self_replacement
            )


    def record_yield(self, code: CodeType, frame: FrameType, yield_value: Any) -> None:
        """Records a yield."""

        # print(f"record_yield {code.co_qualname}")
        if (per_frame := self._pending_traces.get(code)) and (tr := per_frame.get(id(frame))):
            tr.yields.add(get_value_type(yield_value))


    def record_send(self, code: CodeType, frame: FrameType, send_value: Any) -> None:
        """Records a send."""

        # print(f"record_send {code.co_qualname}")
        if (per_frame := self._pending_traces.get(code)) and (tr := per_frame.get(id(frame))):
            tr.sends.add(get_value_type(send_value))


    def _record_variables(self, code: CodeType, frame: FrameType) -> None:
        """Records variables."""
        # Uncomment for debugging: print(f"record_variables {code.co_qualname}")

        if not run_options.variables or not (codevars := code2variables.get(code)):
            return

        # scope_code is guaranteed non-None in code2variables
        if (func_info := self._code2func_info.get(cast_not_None(codevars.scope_code))):
            scope_vars = func_info.variables
        else:
            scope_vars = self._obs.module_variables[Filename(code.co_filename)]

        f_locals = frame.f_locals
        value: Any
        dst: str|None
        for src, dst in codevars.variables.items():
            if (value := f_locals.get(src, NO_OBJECT)) is not NO_OBJECT:
                scope_vars[VariableName(dst)].add(get_value_type(value))

        # Include initial constant types from AST parsing (e.g., x = None).
        # This ensures types from initial assignments aren't lost when
        # the variable is reassigned before function exit.
        for var_name, const_type in codevars.initial_constants.items():
            if (qualified_name := codevars.variables.get(var_name)):
                scope_vars[VariableName(qualified_name)].add(TypeInfo.from_type(const_type))

        if codevars.self and (self_obj := f_locals.get(codevars.self)) is not None:
            obj_attrs = self._object_attributes[codevars.class_key]
            for attr in cast_not_None(codevars.attributes):
                if (value := getattr(self_obj, attr, NO_OBJECT)) is not NO_OBJECT:
                    obj_attrs[VariableName(attr)].add(get_value_type(value))

            # Include initial constant types for attributes (e.g., self.x = None)
            for attr_name, const_type in codevars.attribute_initial_constants.items():
                obj_attrs[VariableName(attr_name)].add(TypeInfo.from_type(const_type))

        # Record class attributes for classmethods (cls.x = ...)
        if codevars.cls and (cls_obj := f_locals.get(codevars.cls)) is not None:
            class_attrs = self._class_attributes[codevars.class_key]
            for attr in codevars.class_attributes or []:
                if (value := getattr(cls_obj, attr, NO_OBJECT)) is not NO_OBJECT:
                    class_attrs[VariableName(attr)].add(get_value_type(value))

            # Include initial constant types for class attributes (e.g., cls.x = None)
            for attr_name, const_type in codevars.class_attribute_initial_constants.items():
                class_attrs[VariableName(attr_name)].add(TypeInfo.from_type(const_type))


    def _record_return_type(self, tr: PendingCallTrace, code: CodeType, ret_type: Any) -> None:
        """Records a pending call trace's return type, finishing the trace."""
        assert tr is not None

        retval_type = (
            ret_type if ret_type is not None
            else (
                # Generators may still be running, or exit with a GeneratorExit exception; we still
                # want them marked as returning None, so they can be simplified to Iterator
                NoneTypeInfo if tr.is_generator else TypeInfo.from_type(typing.NoReturn)
            )
        )

        func_info = self._code2func_info[code]
        func_info.traces.update((tr.finish(retval_type),))


    def record_return(self, code: CodeType, frame: FrameType, return_value: Any) -> bool:
        """Records a return."""

        # print(f"record_return {code.co_qualname}")
        frame_id = id(frame)
        if (per_frame := self._pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code, get_value_type(return_value))
            self._record_variables(code, frame)
            del per_frame[frame_id]
            return True # found it
        else:
            self._record_variables(code, frame)

        return False


    def record_no_return(self, code: CodeType, frame: FrameType) -> bool:
        """Records the lack of a return (e.g., because an exception was raised)."""

        # print(f"record_no_return {code.co_qualname}")
        frame_id = id(frame)
        if (per_frame := self._pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code, None)
            self._record_variables(code, frame)
            del per_frame[frame_id]
            return True # found it
        else:
            self._record_variables(code, frame)

        return False


    def clear_pending(self, code: CodeType) -> None:
        """Discards any pending traces for the given code."""
        self._pending_traces[code].clear()


    def try_close_generators(self) -> None:
        """Attempts to close any generators that may still be running."""
        pending_generators = set()
        for code, per_frame in self._pending_traces.items():
            if any(tr.is_generator for tr in per_frame.values()):
                pending_generators.add(code)

        for code in pending_generators:
            import gc
            for obj in gc.get_referrers(code):
                # In Python 3.13+, close() doesn't generate a PY_UNWIND
                # https://github.com/python/cpython/issues/140373
                if isinstance(obj, GeneratorType):
                    try:
                        obj.throw(GeneratorExit)
                    except:
                        pass

    def _assign_attributes_to_scopes(self) -> None:
        for codevars in code2variables.values():
            if codevars.self:
                assert codevars.class_key is not None
                assert codevars.attributes is not None
                assert codevars.scope_code is not None

                if (func_info := self._code2func_info.get(cast_not_None(codevars.scope_code))):
                    scope_vars = func_info.variables
                else:
                    scope_vars = self._obs.module_variables[Filename(codevars.scope_code.co_filename)]

                obj_attrs = self._object_attributes[codevars.class_key]
                for attr in codevars.attributes:
                    scope_vars[VariableName(f"{codevars.self}.{attr}")] = obj_attrs[VariableName(attr)]

        # Handle class attributes captured via cls.x in classmethods
        # These need to be assigned to the class's scope with qualified names (e.g., "C.monitor")
        for codevars in code2variables.values():
            if codevars.cls and codevars.class_attributes and codevars.class_key:
                class_attrs = self._class_attributes.get(codevars.class_key)
                if class_attrs:
                    # Find the class's CodeVars to get its scope and variable names
                    class_codevars = next(
                        (cv for cv in code2variables.values()
                         if cv.class_name == codevars.class_name and cv.variables),
                        None
                    )
                    if class_codevars and class_codevars.scope_code:
                        if (func_info := self._code2func_info.get(class_codevars.scope_code)):
                            scope_vars = func_info.variables
                        else:
                            scope_vars = self._obs.module_variables[Filename(class_codevars.scope_code.co_filename)]

                        # Use qualified names from the class's variables dict (e.g., "C.monitor")
                        for attr in codevars.class_attributes:
                            if VariableName(attr) in class_attrs:
                                qualified_name = class_codevars.variables.get(attr)
                                if qualified_name:
                                    scope_vars[VariableName(qualified_name)].update(class_attrs[VariableName(attr)])


    def finish_recording(self, main_globals: dict[str, Any]) -> Observations:
        # Any generators left?
        for code, per_frame in self._pending_traces.items():
            for tr in per_frame.values():
                if tr.is_generator:
                    self._record_return_type(tr, code, None)

        self._assign_attributes_to_scopes()

        obs, self._obs = self._obs, Observations()
        self._code2func_info.clear()
        self._pending_traces.clear()
        self._object_attributes.clear()
        self._class_attributes.clear()

        # The type map depends on main_globals as well as the on the state
        # of sys.modules, so we can't postpone them until collect_annotations,
        # which operates on deserialized data (vs. data just collected).
        type_map = TypeMap(main_globals)

        if run_options.adjust_type_names:
            type_name_adjuster = AdjustTypeNamesT(type_map)
            obs.transform_types(type_name_adjuster)
        else:
            type_name_adjuster = None
            obs.transform_types(CheckTypeNamesT(type_map))

        if run_options.resolve_mocks:
            obs.transform_types(ResolveMocksT(type_name_adjuster))

        obs.test_modules = set(run_options.test_modules) | set(detected_test_modules) | {
            mod_name
            for mod_name in obs.source_to_module_name.values()
            if is_test_module(mod_name)
        }

        if run_options.exclude_test_files:
            # should_skip_function doesn't know to skip test files until they are detected,
            # so we can't help but get events for test modules while they are being loaded.
            for f in obs.source_to_module_name.keys() & detected_test_files:
                del obs.source_to_module_name[f]

        return obs


def get_self_type(
    code,
    arg_info: inspect.ArgInfo
) -> tuple[TypeInfo|None, TypeInfo|None, OverriddenFunction|None]:
    if arg_info.args:
        first_arg = arg_info.locals[arg_info.args[0]]

        name = code.co_name
        if (
            name.startswith("__")
            and not name.endswith("__")
            and len(parts := code.co_qualname.split(".")) > 1
        ):
            # parts[-2] may be "<locals>"... that's ok, as we then have
            # a local function and there is no 'Self' to find.
            name = f"_{parts[-2]}{name}"    # private attribute/method

        # if type(first_arg) is type, we may have a @classmethod
        first_arg_class = first_arg if type(first_arg) is type else type(first_arg)

        # @property?
        is_property = isinstance(getattr(type(first_arg), name, None), property)

        # find class that defines that name, in case it's inherited
        defining_class, next_index = next(
            (
                (ancestor, i+1)
                for i, ancestor in enumerate(first_arg_class.__mro__)
                if (
                    (is_property and name in ancestor.__dict__)
                    or (
                        (f := unwrap(ancestor.__dict__.get(name, None)))
                        and getattr(f, "__code__", None) is code
                    )
                )
            ),
            (None, None)
        )

        if not defining_class:
            return None, None, None

        # The first argument is 'Self' and the type of 'Self', in the context of
        # its definition, is "defining_class"; now let's see if this method
        # overrides another
        overrides = None
        if not (
            is_property
            or name in ('__init__', '__new__')  # irrelevant for Liskov
        ):
            overrides = next(
                (
                    OverriddenFunction(
                        # wrapper_descriptor and possibly other native objects may lack __module__
                        normalize_module_name(getattr(f, "__module__", ancestor.__module__)),
                        f.__qualname__,
                        CodeId.from_code(f.__code__) if hasattr(f, "__code__") else None,
                        get_parent_arg_types(f, arg_info)
                    )
                    for ancestor in first_arg_class.__mro__[next_index:]
                    if (f := unwrap(ancestor.__dict__.get(name, None)))
                    if getattr(f, "__code__", None) is not code
                ),
                None
            )

        return get_type_name(first_arg_class), get_type_name(defining_class), overrides

    return None, None, None


def get_defaults(code, frame) -> dict[str, TypeInfo]:
    if (function := find_function(frame, code)):
        return {
            param_name: get_value_type(param.default)
            for param_name, param in inspect.signature(function).parameters.items()
            if param.default is not param.empty
        }

    return {}


def get_parent_arg_types(
    parents_func: object,
    child_arg_info: inspect.ArgInfo
) -> tuple[TypeInfo|None, ...] | None:
    """Returns inline type annotations for a parent's method's arguments."""

    if not (
        isinstance(parents_func, FunctionType) 
        and (co := getattr(parents_func, "__code__", None))
    ):
        return None

    try:
        parent_args = inspect.getargs(co)
        if not (hints := typing.get_type_hints(parents_func)):
            return None
    except (NameError, TypeError) as e:
        logger.info(f"Error getting args or type hints for {parents_func} " +
                    f"({parents_func.__annotations__}): {e}.\n")
        return None

    result = [
        # First the positional, looking up by their names given in the parent.
        # Note that for the override to be valid, their signatures must have
        # the same number of positional arguments.
        hint2type(hints[arg]) if arg in hints else None
        for arg in co.co_varnames[:co.co_argcount]
    ] + [
        # Then kwonly, going by the order (and quantity) in the child
        hint2type(hints[arg]) if arg in hints else None
        for arg in child_arg_info.args[co.co_argcount:]
    ]

    # Then varargs and varkw, if they exist.  Note that for the override to
    # be valid, both must agree to include or not those arguments (but their
    # names may change)
    if parent_args.varargs:
        result += [
            hint2type(hint) if (hint := hints.get(parent_args.varargs)) else None
        ]

    if parent_args.varkw:
        result += [
            hint2type(hint) if (hint := hints.get(parent_args.varkw)) else None
        ]

    return tuple(result)


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
