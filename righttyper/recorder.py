import inspect
import builtins
from dataclasses import dataclass, field
from types import CodeType, FrameType, FunctionType, GeneratorType
from collections import defaultdict
import collections.abc as abc
from pathlib import Path
import logging
from righttyper.logger import logger
from righttyper.righttyper_types import ArgumentName, VariableName, Filename, CodeId, CallableWithCode, cast_not_None
from righttyper.typeinfo import TypeInfo, NoneTypeInfo, UnknownTypeInfo, CallTrace
from typing import Final, Any, NewType, overload
import typing
from righttyper.observations import Observations, FuncInfo, OverriddenFunction, ArgInfo
from righttyper.variable_capture import code2variables, Constructor
from righttyper.options import run_options, output_options
from righttyper.righttyper_utils import (
    source_to_module_fqn, get_main_module_fqn, skip_this_file,
    detected_test_files, detected_test_modules, is_test_module,
    normalize_module_name, unwrap
)
from righttyper.type_id import find_function, get_value_type, get_type_name, hint2type
from righttyper.righttyper_tool import field_class_init_codes
from righttyper.typemap import TypeMap, AdjustTypeNamesT, CheckTypeNamesT


# Singleton used to differentiate from None
NO_OBJECT: Final = object()


# Synthetic key in `func_info.constructor_types` used to carry the resolved
# return-expression constructor type.  `return` is a Python keyword, so it
# cannot collide with any user-defined variable name.  mk_annotation
# extracts this and applies it post-merge to the retval annotation so
# `def f(): p = Path(...); return p` widens consistently (both `p` and the
# function's return become Path, not PosixPath).
RETURN_CONSTRUCTOR_KEY: Final = VariableName("return")


def _walk_constructor_callee(
    parts: list[str],
    f_locals: abc.Mapping[str, Any],
    f_globals: abc.Mapping[str, Any],
) -> tuple[type, str | None] | None:
    """Walk the dotted callee parts (`["Path"]`, `["pathlib", "Path"]`,
    `["Path", "cwd"]`) against the live frame.  Locals are tried first so
    local aliases like `P = Path; p = P(...)` resolve correctly; globals
    cover imports.  Builtins (`set`, `list`, `dict`, ...) are not looked up
    — for them, ceiling and observation share the same `type_obj` and the
    post-merge widening is a no-op anyway.

    Returns `(last_class, method_name)` where `last_class` is the most
    recent class encountered on the walk and `method_name` is None if the
    final target *is* `last_class` (direct constructor call) or the last
    part name if the final target is something else (a method/function on
    the class — factory pattern).  Returns None for shapes that don't fit
    either pattern (e.g. free function `os.getcwd`).

    The actual constructor type, including typeshed lookup for factory
    cases, is determined later in finish_recording (when TypeMap is
    available)."""
    from righttyper.random_dict import RandomDict
    obj = f_locals.get(parts[0], NO_OBJECT)
    if obj is NO_OBJECT:
        obj = f_globals.get(parts[0], NO_OBJECT)
    if obj is NO_OBJECT or obj is None:
        return None
    last_class = obj if isinstance(obj, type) else None
    for p in parts[1:]:
        nxt = getattr(obj, p, NO_OBJECT)
        if nxt is NO_OBJECT or nxt is None:
            return None
        obj = nxt
        if isinstance(obj, type):
            last_class = obj
    if last_class is None or last_class is RandomDict:
        # RandomDict is RightTyper's internal stand-in for dict (under
        # --replace-dict).  type_id._handle_randomdict masquerades it as
        # plain `dict` for type-reporting; surfacing it as a ceiling
        # would introduce RandomDict into user output.  Drop here.
        return None
    method_name = None if isinstance(obj, type) else parts[-1]
    return (last_class, method_name)


def _get_field_names(cls: type) -> tuple[str, ...]|None:
    """Returns field names for dataclasses, attrs, and NamedTuple classes, or None."""
    if hasattr(cls, '__dataclass_fields__'):
        return tuple(cls.__dataclass_fields__)
    if attrs := getattr(cls, '__attrs_attrs__', None):
        return tuple(a.name for a in attrs)
    if issubclass(cls, tuple) and hasattr(cls, '_fields'):
        return cls._fields  # type: ignore[union-attr]
    return None




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
    ) -> None:
        self.arg_info = arg_info
        self.args_start = self._get_arg_types(arg_info)
        self.yields: set[TypeInfo] = set()
        self.sends: set[TypeInfo] = set()
        self.is_async = bool(co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE))
        self.is_generator=bool(co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR))
        self.self_type = self_type


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

        return CallTrace(type_data, first_arg_class=self.self_type)


class ObservationsRecorder:
    def __init__(self) -> None:
        # Finds FuncInfo by their CodeType
        self._code2func_info: dict[CodeType, FuncInfo] = {}

        # Record-time staging for constructor ceilings.  At record time we
        # walk the live frame to resolve `var = Foo(...)` / `Class.method(...)`
        # shapes to a (last_class, method_name) tuple — handles local
        # aliases.  Final canonicalization and typeshed Self-lookup happen
        # in finish_recording, once TypeMap is built.  Two parallel dicts
        # mirror the function-level vs module-level split of scope_vars.
        self._func_var_callees: dict[
            CodeType, dict[VariableName, set[tuple[type, str | None]]]
        ] = defaultdict(lambda: defaultdict(set))
        self._module_var_callees: dict[
            Filename, dict[VariableName, set[tuple[type, str | None]]]
        ] = defaultdict(lambda: defaultdict(set))

        # Started, but not (yet) completed traces
        self._pending_traces: dict[CodeType, dict[FrameId, PendingCallTrace]] = defaultdict(dict)

        # Pending wrapped function traces, keyed by wrapped function's code and
        # the wrapper's frame id.  Created when a wrapper starts; completed when
        # the wrapper returns.
        self._pending_wrapped_traces: dict[CodeType, dict[FrameId, PendingCallTrace]] = defaultdict(dict)

        # Field-class __init__/__new__ codes already seen; used to clear stale class-body types
        self._field_classes_seen: set[CodeType] = set()

        # Object attributes: class_key -> attr_name -> set[TypeInfo]
        self._object_attributes: dict[object, dict[VariableName, set[TypeInfo]]] = defaultdict(lambda: defaultdict(set))

        # Class attributes: class_key -> attr_name -> set[TypeInfo]
        self._class_attributes: dict[object, dict[VariableName, set[TypeInfo]]] = defaultdict(lambda: defaultdict(set))

        self._obs = Observations()


    def past_warmup(self, code: CodeType) -> bool:
        """Returns True if we've collected enough samples to switch to Poisson timing."""
        if (func_info := self._code2func_info.get(code)):
            return func_info.traces.total() >= run_options.poisson_warmup_samples
        return False


    def _register_module(self, filename: str, modname: str | None) -> None:
        """Registers a module's filename-to-name mapping if not already known."""
        if filename and filename not in self._obs.source_to_module_name:
            if modname == "__main__":
                modname = get_main_module_fqn()
            if not modname:
                modname = source_to_module_fqn(Path(filename))
            if modname:
                self._obs.source_to_module_name[Filename(filename)] = modname

    def record_module(
        self,
        code: CodeType,
        frame: FrameType
    ) -> None:
        # TODO handle cases where modules are loaded more than once, e.g. through pytest
        self._register_module(code.co_filename, frame.f_globals.get('__name__', None))


    def _register_function(
        self,
        code: CodeType,
        function: CallableWithCode|None,
        arg_info: inspect.ArgInfo,
        overrides: list[OverriddenFunction],
        defining_class: TypeInfo|None = None,
    ) -> None:
        """Registers a function if not already known."""

        if code not in self._code2func_info:
            arg_names = (
                *(a for a in arg_info.args),
                *((arg_info.varargs,) if arg_info.varargs else ()),
                *((arg_info.keywords,) if arg_info.keywords else ())
            )

            defaults: dict[str, TypeInfo] = {}
            if function:
                try:
                    for param_name, param in inspect.signature(function).parameters.items():
                        if param.default is not param.empty:
                            defaults[param_name] = get_value_type(param.default)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Unable to get signature for {code.co_qualname}: {e}")

            # Use resolved CodeId for dataclass/attrs/NamedTuple __init__
            cls_info = field_class_init_codes.get(code)
            if cls_info is not None:
                cls, code_id = cls_info
                # Register the real source file so process() finds it
                if code_id.file_name not in self._obs.source_to_module_name:
                    self._obs.source_to_module_name[code_id.file_name] = cls.__module__
            else:
                code_id = CodeId.from_code(code)

            self._code2func_info[code] = func_info = FuncInfo(
                code_id,
                tuple(
                    ArgInfo(ArgumentName(name), defaults.get(name))
                    for name in arg_names
                ),
                ArgumentName(arg_info.varargs) if arg_info.varargs else None,
                ArgumentName(arg_info.keywords) if arg_info.keywords else None,
                overrides=overrides,
                defining_class=defining_class,
            )
            self._obs.func_info[func_info.code_id] = func_info


    def _register_parent_function(
        self, child_fi: FuncInfo, parent_func: CallableWithCode, finder: 'OverrideFinder',
        parent_defining_class: TypeInfo | None,
    ) -> None:
        """Registers all ancestor functions that the child overrides.

        Walks the remaining MRO via the OverrideFinder.  Each ancestor is registered
        independently and added to child_fi.overrides — no parent→grandparent chain
        is created, which avoids cross-contamination between sibling parents in
        multiple inheritance.
        """
        while True:
            parent_code = parent_func.__code__

            if parent_code in self._code2func_info:
                return  # already registered (with its own overrides)

            if skip_this_file(parent_code.co_filename):
                return

            self._register_module(parent_code.co_filename, getattr(parent_func, '__module__', None))

            parent_args = inspect.getargs(parent_code)
            parent_arg_info = inspect.ArgInfo(parent_args.args, parent_args.varargs, parent_args.varkw, {})

            # Register this parent with no overrides of its own
            self._register_function(parent_code, parent_func, parent_arg_info, [],
                                    defining_class=parent_defining_class)
            self._code2func_info[parent_code].is_abstract = getattr(parent_func, '__isabstractmethod__', False)

            # Find next ancestor in the child's MRO
            result = finder.find_next(parent_code, parent_arg_info)
            if not result:
                return

            override, next_func, next_class = result
            child_fi.overrides.append(override)

            if not next_func:
                return

            parent_func = next_func
            parent_defining_class = (
                get_type_name(next_class) if next_class is not None else None
            )


    def record_start(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo
    ) -> None:
        """Records a function start."""

        # print(f"record_start {code.co_qualname} {arg_types}")

        # Skip record_module for synthetic __init__/__new__ (dataclass, attrs,
        # NamedTuple) whose co_filename is '<string>' or similar — their real
        # source file is registered in record_function.  User-defined __init__
        # has a real co_filename and needs record_module.
        co_fn = code.co_filename
        if code not in field_class_init_codes or not (co_fn.startswith('<') and co_fn.endswith('>')):
            self.record_module(code, frame)

        # CO_NEWLOCALS, when set within a CodeType's co_flags, indicates that
        # a new "locals" dictionary is created; it roughly indicates a function
        # call, as the new dictionary is created for the new scope.
        if (code.co_flags & inspect.CO_NEWLOCALS):
            sti = get_self_type(code, arg_info)
            self._register_function(code, find_function(frame, code), arg_info,
                                    sti.overrides, defining_class=sti.self_replacement)

            if sti.parent_func is not None and sti.override_finder is not None:
                child_fi = self._code2func_info[code]
                self._register_parent_function(child_fi, sti.parent_func,
                                               sti.override_finder, sti.parent_defining_class)

            self._pending_traces[code][id(frame)] = PendingCallTrace(
                arg_info, code.co_flags, sti.self_type
            )

            if run_options.propagate_wrapped_types:
                self._record_wrapped_function_types(code, frame, arg_info)


    def _record_wrapped_function_types(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo
    ) -> None:
        """If 'code' is a wrapper with __wrapped__, creates a pending trace for the wrapped function."""
        from righttyper.righttyper_tool import wrapped_by

        wrapped = wrapped_by.get(code)
        if not wrapped:
            return
        wrapped_code = wrapped.__code__

        # Mark this wrapper function to skip annotation
        self._obs.wrapper_code_ids.add(CodeId.from_code(code))

        # Register the wrapped function's module
        self._register_module(wrapped_code.co_filename, getattr(wrapped, '__module__', None))

        # Extract actual positional and keyword args from the wrapper's frame
        f_locals = frame.f_locals

        # Determine if the wrapper is a method by checking if its code object
        # is defined in the first argument's type hierarchy.
        skip_first = (
            arg_info.args
            and arg_info.args[0] in f_locals
            and is_method_of(code, f_locals[arg_info.args[0]])
        )
        wrapper_args = arg_info.args[1:] if skip_first else arg_info.args

        if arg_info.varargs and arg_info.varargs in f_locals:
            actual_positional = tuple(f_locals[arg_info.varargs])
        else:
            actual_positional = tuple(
                f_locals[a] for a in wrapper_args if a in f_locals
            )

        if arg_info.keywords and arg_info.keywords in f_locals:
            actual_keywords: dict[str, Any] = dict(f_locals[arg_info.keywords])
        else:
            actual_keywords = {}

        # Capture keyword-only wrapper args (in f_locals but not in *args or **kwargs)
        if arg_info.varargs:
            for name in wrapper_args:
                if name not in actual_keywords and name in f_locals:
                    actual_keywords[name] = f_locals[name]

        # Map wrapper args to the wrapped function's parameters
        try:
            bound = inspect.signature(wrapped).bind_partial(*actual_positional, **actual_keywords)
            bound.apply_defaults()
            synthetic_locals = dict(bound.arguments)
        except (TypeError, ValueError) as e:
            logger.debug(f"wrapped function {wrapped_code.co_qualname}: "
                         f"failed to bind args: {e}")
            return

        wrapped_args = inspect.getargs(wrapped_code)

        synthetic_arg_info = inspect.ArgInfo(
            wrapped_args.args, wrapped_args.varargs, wrapped_args.varkw,
            synthetic_locals
        )

        sti = get_self_type(wrapped_code, synthetic_arg_info)

        # Register the wrapped function if not already known
        self._register_function(wrapped_code, wrapped, synthetic_arg_info,
                                sti.overrides, defining_class=sti.self_replacement)

        pending = PendingCallTrace(synthetic_arg_info, wrapped_code.co_flags, sti.self_type)
        self._pending_wrapped_traces[wrapped_code][id(frame)] = pending


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

        # Field-class __init__/__new__: inspect instance to get field types.
        if (cls_info := field_class_init_codes.get(code)) is not None:
            cls, code_id = cls_info
            field_names = _get_field_names(cls)
            if field_names is not None:
                # NamedTuple uses __new__ with fields as local vars; others use __init__ with self
                is_namedtuple = issubclass(cls, tuple)
                f_locals = frame.f_locals
                if is_namedtuple or f_locals.get("self") is not None:
                    module_vars = self._obs.module_variables[code_id.file_name]
                    qualname = cls.__qualname__
                    # First call: clear stale types from class-body capture (e.g. Field descriptors)
                    first_call = code not in self._field_classes_seen
                    if first_call:
                        self._field_classes_seen.add(code)
                    for field_name in field_names:
                        if is_namedtuple:
                            value = f_locals.get(field_name, NO_OBJECT)
                        else:
                            value = getattr(f_locals["self"], field_name, NO_OBJECT)
                        if value is not NO_OBJECT:
                            key = VariableName(f"{qualname}.{field_name}")
                            if first_call:
                                module_vars[key] = {get_value_type(value)}
                            else:
                                module_vars[key].add(get_value_type(value))

        if not run_options.variables or not (codevars := code2variables.get(code)):
            return

        # scope_code is guaranteed non-None in code2variables
        scope_callees: dict[VariableName, set[tuple[type, str | None]]]
        if (func_info := self._code2func_info.get(cast_not_None(codevars.scope_code))):
            scope_vars = func_info.variables
            scope_callees = self._func_var_callees[cast_not_None(codevars.scope_code)]
        else:
            scope_vars = self._obs.module_variables[Filename(code.co_filename)]
            scope_callees = self._module_var_callees[Filename(code.co_filename)]

        f_locals = frame.f_locals
        prefix = codevars.var_prefix

        for var_name, init in codevars.variables.items():
            qualified = f"{prefix}{var_name}"
            # Record runtime value
            if (value := f_locals.get(var_name, NO_OBJECT)) is not NO_OBJECT:
                scope_vars[VariableName(qualified)].add(get_value_type(value))
            # Include initial constant type if present (e.g., x = None)
            if isinstance(init, type):
                scope_vars[VariableName(qualified)].add(TypeInfo.from_type(init))
            elif isinstance(init, Constructor):
                # Resolve the source-level dotted name against the live frame
                # — locals first (for `P = Path; p = P(...)`), then globals.
                # The result is staged; canonicalization (private-module
                # paths like Python 3.13's `pathlib._local.Path`) and the
                # typeshed Self lookup happen in finish_recording, where
                # TypeMap is available.
                if (callee := _walk_constructor_callee(
                        init.parts, f_locals, frame.f_globals)) is not None:
                    scope_callees[VariableName(qualified)].add(callee)

        # Same idea for the function's return: stage the callee under the
        # synthetic <return> key; finalized in finish_recording.
        if codevars.return_constructor is not None:
            if (callee := _walk_constructor_callee(
                    codevars.return_constructor.parts,
                    f_locals, frame.f_globals)) is not None:
                scope_callees[RETURN_CONSTRUCTOR_KEY].add(callee)

        if codevars.self and (self_obj := f_locals.get(codevars.self)) is not None:
            obj_attrs = self._object_attributes[codevars.class_key]
            for attr, const_type in cast_not_None(codevars.attributes).items():
                if (value := getattr(self_obj, attr, NO_OBJECT)) is not NO_OBJECT:
                    obj_attrs[VariableName(attr)].add(get_value_type(value))
                # Include initial constant type if present (e.g., self.x = None)
                if const_type is not None:
                    obj_attrs[VariableName(attr)].add(TypeInfo.from_type(const_type))

        # Record class attributes for classmethods (cls.x = ...)
        if codevars.cls and (cls_obj := f_locals.get(codevars.cls)) is not None:
            class_attrs = self._class_attributes[codevars.class_key]
            for attr, const_type in (codevars.class_attributes or {}).items():
                if (value := getattr(cls_obj, attr, NO_OBJECT)) is not NO_OBJECT:
                    class_attrs[VariableName(attr)].add(get_value_type(value))
                # Include initial constant type if present (e.g., cls.x = None)
                if const_type is not None:
                    class_attrs[VariableName(attr)].add(TypeInfo.from_type(const_type))


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


    def _complete_wrapped_trace(self, code: CodeType, frame_id: FrameId, return_value: Any) -> None:
        """Completes any pending wrapped trace when a wrapper returns."""
        from righttyper.righttyper_tool import wrapped_by

        wrapped = wrapped_by.get(code)
        if not wrapped:
            return
        wrapped_code = wrapped.__code__

        if (per_frame := self._pending_wrapped_traces.get(wrapped_code)):
            if (tr := per_frame.pop(frame_id, None)):
                retval_type = (
                    get_value_type(return_value)
                    if run_options.infer_wrapped_return_type
                    else UnknownTypeInfo
                )
                self._record_return_type(tr, wrapped_code, retval_type)


    def record_return(self, code: CodeType, frame: FrameType, return_value: Any) -> bool:
        """Records a return."""

        # print(f"record_return {code.co_qualname}")
        frame_id = id(frame)
        if (per_frame := self._pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code, get_value_type(return_value))
            self._record_variables(code, frame)
            del per_frame[frame_id]
            self._complete_wrapped_trace(code, frame_id, return_value)
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
            # Discard wrapped trace on exception
            self._discard_wrapped_trace(code, frame_id)
            return True # found it
        else:
            self._record_variables(code, frame)

        return False


    def _discard_wrapped_trace(self, code: CodeType, frame_id: FrameId) -> None:
        """Discards any pending wrapped trace when a wrapper raises."""
        from righttyper.righttyper_tool import wrapped_by

        wrapped = wrapped_by.get(code)
        if not wrapped:
            return

        if (per_frame := self._pending_wrapped_traces.get(wrapped.__code__)):
            per_frame.pop(frame_id, None)


    def clear_pending(self, code: CodeType) -> None:
        """Discards any pending traces for the given code."""
        self._pending_traces[code].clear()
        # Also clear any wrapped traces triggered by this wrapper
        from righttyper.righttyper_tool import wrapped_by
        wrapped = wrapped_by.get(code)
        if wrapped:
            self._pending_wrapped_traces[wrapped.__code__].clear()


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

                        # Use var_prefix to build qualified names (e.g., "C.monitor")
                        for attr in codevars.class_attributes:
                            if VariableName(attr) in class_attrs:
                                qualified_name = f"{class_codevars.var_prefix}{attr}"
                                scope_vars[VariableName(qualified_name)].update(class_attrs[VariableName(attr)])


    def _finalize_constructor_types(
        self,
        var_callees: dict[VariableName, set[tuple[type, str | None]]],
        out: "abc.MutableMapping[VariableName, set[TypeInfo]]",
        type_map: TypeMap,
    ) -> None:
        """Finalize staged constructor-type candidates: canonicalize the
        resolved class via TypeMap and consult typeshed for the callee's
        declared return type (`__new__` for direct constructor calls, the
        named method for factory calls).  Write the resolved TypeInfo into
        `out[var_name]` — typically `func_info.constructor_types[var_name]`
        or `obs.module_constructor_types[filename][var_name]`.

        The runtime-vs-static consistency check (whether the observed
        runtime type is actually a strict subclass of the constructor
        type) lives at mk_annotation time, where retval observations from
        traces are available alongside per-variable observations."""
        from righttyper.typeshed import get_typeshed_func_return
        for var_name, callees in var_callees.items():
            for last_class, method_name in callees:
                names = type_map.find(last_class)
                if not names:
                    continue
                module, qualname = names[0]
                lookup = method_name if method_name is not None else '__new__'
                ret = get_typeshed_func_return(
                    module if module else 'builtins', f"{qualname}.{lookup}")
                if ret is None:
                    # No typeshed entry: for direct calls fall back to
                    # last_class (most classes inherit object.__new__,
                    # which is `Self`).  For factory calls we don't know
                    # what's returned, so skip.
                    if method_name is not None:
                        continue
                elif (
                    (ret.module in ('typing', 'typing_extensions') and ret.name == 'Self')
                    or (ret.module == module and ret.name == qualname)
                ):
                    # Self (from typing / typing_extensions), or the class
                    # explicitly named — both resolve to last_class.
                    pass
                else:
                    # TODO resolve specific non-Self return types — needs
                    # a sys.modules lookup, parametrization-loss handling,
                    # and TypeVar substitution.
                    continue
                out[var_name].add(get_type_name(last_class))


    def finish_recording(self, main_globals: dict[str, Any]) -> Observations:
        # Any generators left?
        for code, per_frame in self._pending_traces.items():
            for tr in per_frame.values():
                if tr.is_generator:
                    self._record_return_type(tr, code, None)

        self._assign_attributes_to_scopes()

        # Populate accessed_attributes from the loader's static analysis.
        # Done here (vs. at collect_annotations time) so the data survives
        # serialization to .rt files for the --only-collect + process flow.
        # co_varnames/co_freevars are only available while the code object
        # is live, so module_accessed_attributes must be computed now.
        # Skipped when --no-use-attribute-simplification: the loader will not
        # have populated cv.accessed_attributes either, but the dict-write
        # overhead is still worth avoiding.
        if output_options.use_attribute_simplification:
            for co, cv in code2variables.items():
                if cv.accessed_attributes:
                    self._obs.accessed_attributes[CodeId.from_code(co)] = cv.accessed_attributes
                    module_attrs = self._obs.module_accessed_attributes.setdefault(
                        Filename(co.co_filename), {}
                    )
                    for var_name, attrs in cv.accessed_attributes.items():
                        if var_name not in co.co_varnames and var_name not in co.co_freevars:
                            module_attrs.setdefault(var_name, set()).update(attrs)

        obs, self._obs = self._obs, Observations()
        self._code2func_info.clear()
        self._pending_traces.clear()
        self._pending_wrapped_traces.clear()
        self._object_attributes.clear()
        self._class_attributes.clear()

        # The type map depends on main_globals as well as the on the state
        # of sys.modules, so we can't postpone them until collect_annotations,
        # which operates on deserialized data (vs. data just collected).
        type_map = TypeMap(main_globals)

        # Finalize constructor types staged at record time *before*
        # AdjustTypeNamesT runs, so the new entries get canonicalized
        # alongside everything else in obs (avoids leaking private
        # __module__ paths like `_io` or `pathlib._local`).  type_obj is
        # set on each new entry for the canonicalization pass; it's
        # cleared at .rt save by the Pickler dispatch_table.  `obs` is the
        # swapped-out Observations; its func_info is keyed by CodeId.
        for code, var_callees in self._func_var_callees.items():
            if (fi := obs.func_info.get(CodeId.from_code(code))):
                self._finalize_constructor_types(
                    var_callees, fi.constructor_types, type_map)
        for filename, var_callees in self._module_var_callees.items():
            self._finalize_constructor_types(
                var_callees, obs.module_constructor_types[filename], type_map)

        if run_options.adjust_type_names:
            type_name_adjuster = AdjustTypeNamesT(type_map)
            obs.transform_types(type_name_adjuster)
            obs.type_name_map = type_map.to_name_map()
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

        # If the runner script lives outside --root, treat its module as a
        # test module: any classes defined in the runner are scaffolding,
        # not part of any package's contract.
        main_file = main_globals.get('__file__') if main_globals else None
        if main_file and run_options.script_dir:
            script_dir = Path(run_options.script_dir).resolve()
            main_path = Path(main_file).resolve()
            if not main_path.is_relative_to(script_dir):
                if (
                    (main_spec := main_globals.get('__spec__'))
                    and main_spec.origin
                    and (main_name := source_to_module_fqn(Path(main_spec.origin)))
                ):
                    obs.test_modules.add(main_name)
                else:
                    obs.test_modules.add('__main__')

        if run_options.exclude_test_files:
            # should_skip_function doesn't know to skip test files until they are detected,
            # so we can't help but get events for test modules while they are being loaded.
            for f in obs.source_to_module_name.keys() & detected_test_files:
                del obs.source_to_module_name[f]

        return obs


@dataclass
class MethodInfo:
    """Information about a method's definition in a class hierarchy."""
    first_arg_class: type       # The type of the first argument (or the type itself for classmethod)
    defining_class: type        # The class where this method is defined
    next_index: int             # Index into MRO for finding overridden methods
    name: str                   # The method name (possibly mangled for private methods)
    is_property: bool           # Whether this is a property


def find_method_info(code: CodeType, first_arg: object) -> MethodInfo | None:
    """Finds method information for 'code' in the type hierarchy of 'first_arg'.

    Returns MethodInfo if 'code' is a method defined on first_arg's type, None otherwise.
    """
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
    result = next(
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
        None
    )

    if result is None:
        return None

    defining_class, next_index = result
    return MethodInfo(first_arg_class, defining_class, next_index, name, is_property)


def is_method_of(code: CodeType, first_arg: object) -> bool:
    """Returns True if 'code' is a method defined on the type of 'first_arg'."""
    return find_method_info(code, first_arg) is not None


class OverrideFinder:
    """Walks a class MRO to find method overrides.

    Tracks the current position in the MRO so that successive calls to find_next()
    walk further up the hierarchy (child → parent → grandparent → ...).
    """

    def __init__(self, mro: tuple[type, ...], method_name: str, start_index: int):
        self._mro = mro
        self._method_name = method_name
        self._index = start_index

    def find_next(
        self, code: CodeType, child_arg_info: inspect.ArgInfo
    ) -> tuple[OverriddenFunction, FunctionType | None, type | None] | None:
        """Find the next override of the method in the MRO.

        Returns (overrides, parent_func, defining_class) or None.
        parent_func is None for native/builtin methods.
        defining_class is the MRO ancestor that defines the override.
        """
        for idx, ancestor in enumerate(self._mro[self._index:]):
            f = unwrap(ancestor.__dict__.get(self._method_name, None))
            if f and getattr(f, "__code__", None) is not code:
                self._index = self._index + idx + 1
                return (
                    OverriddenFunction(
                        normalize_module_name(getattr(f, "__module__", ancestor.__module__)),
                        f.__qualname__,
                        CodeId.from_code(f.__code__) if hasattr(f, "__code__") else None,
                        get_parent_arg_types(f, child_arg_info)
                    ),
                    f if isinstance(f, FunctionType) else None,
                    ancestor,
                )
        return None


@dataclass
class SelfTypeInfo:
    """Result of get_self_type: information about a method's self/cls and override relationship."""
    self_type: TypeInfo | None = None
    self_replacement: TypeInfo | None = None
    overrides: list[OverriddenFunction] = field(default_factory=list)
    # Typed as CallableWithCode (a Protocol) rather than FunctionType to avoid
    # typeshed's descriptor-protocol behavior on instance access — accessing a
    # FunctionType-typed dataclass field via an instance reports MethodType.
    parent_func: CallableWithCode | None = None
    parent_defining_class: TypeInfo | None = None  # defining class of parent_func
    override_finder: OverrideFinder | None = None  # for walking the parent chain


_NO_SELF_TYPE_INFO = SelfTypeInfo()


def get_self_type(
    code: CodeType,
    arg_info: inspect.ArgInfo
) -> SelfTypeInfo:
    if arg_info.args:
        first_arg = arg_info.locals[arg_info.args[0]]

        info = find_method_info(code, first_arg)
        if not info:
            return _NO_SELF_TYPE_INFO

        # The first argument is 'Self' and the type of 'Self', in the context of
        # its definition, is "defining_class"; now let's see if this method
        # overrides another
        overrides: list[OverriddenFunction] = []
        parent_func = None
        finder: OverrideFinder | None = None
        parent_defining_class: TypeInfo | None = None
        if not (
            info.is_property
            or info.name in ('__init__', '__new__')  # irrelevant for Liskov
        ):
            finder = OverrideFinder(info.first_arg_class.__mro__, info.name, info.next_index)
            result = finder.find_next(code, arg_info)
            if result:
                overrides = [result[0]]
                parent_func = result[1]
                parent_defining_class = (
                    get_type_name(result[2]) if result[2] is not None else None
                )

        return SelfTypeInfo(
            get_type_name(info.first_arg_class),
            get_type_name(info.defining_class),
            overrides,
            parent_func,
            parent_defining_class=parent_defining_class,
            override_finder=finder if parent_func else None,
        )

    return _NO_SELF_TYPE_INFO



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
