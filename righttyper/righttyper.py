import ast
import concurrent.futures
import importlib.metadata
import importlib.util
import inspect
import functools
import logging
import os
import runpy
import signal
import sys
import platform
import typeshed_client as ts
import pickle
import datetime
import json
import subprocess
import re
import time
import builtins


import collections.abc as abc
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import CodeType, FrameType, FunctionType, MethodType, GeneratorType, AsyncGeneratorType, UnionType
import typing
from typing import (
    Any,
    TextIO,
    Self,
    Sequence,
    overload
)

import click

from righttyper.righttyper_process import (
    process_file,
    SignatureChanges
)
from righttyper.righttyper_runtime import (
    find_function,
    unwrap,
    get_value_type,
    get_type_name,
    hint2type,
    PostponedIteratorArg,
)
from righttyper.righttyper_tool import (
    TOOL_ID,
    TOOL_NAME,
    register_monitoring_callbacks,
    reset_monitoring
)
from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    CodeId,
    Filename,
    FuncId,
    FuncInfo,
    FrameId,
    FuncAnnotation,
    FunctionName,
    FunctionDescriptor,
    TypeInfo,
    NoneTypeInfo,
    AnyTypeInfo,
    CallTrace,
    PendingCallTrace,
    UnknownTypeInfo
)
from righttyper.typeinfo import (
    merged_types,
    generalize,
)
from righttyper.righttyper_utils import (
    skip_this_file,
    should_skip_function,
    detected_test_modules,
    is_test_module,
    source_to_module_fqn,
    get_main_module_fqn,
)
import righttyper.loader as loader
from righttyper.righttyper_alarm import (
    SignalAlarm,
    ThreadAlarm,
)

from righttyper.options import Options, options
from righttyper.logger import logger
from righttyper.typemap import AdjustTypeNamesT
from righttyper.atomic import AtomicCounter

PKL_FILE_NAME = f"{TOOL_NAME}.rt"
PKL_FILE_VERSION = 4


# Overloads so we don't have to always write CodeId(id(code)), etc.
@overload
def id(obj: CodeType) -> CodeId: ...
@overload
def id(obj: FrameType) -> FrameId: ...
@overload
def id(obj: object) -> int: ...
def id(obj):
    return builtins.id(obj)


def get_inline_arg_types(
    parents_func: FunctionType|FunctionDescriptor,
    child_args: tuple[ArgInfo, ...]
) -> list[TypeInfo|None] | None:
    """Returns inline type annotations for a parent's method's arguments."""

    if not (co := getattr(parents_func, "__code__", None)):
        return None

    try:
        if not (hints := typing.get_type_hints(parents_func)):
            return None
    except (NameError, TypeError) as e:
        logger.info(f"Error getting type hints for {parents_func} " + 
                    f"({parents_func.__annotations__}): {e}.\n")
        return None

    return (
        # first the positional, looking up by their names given in the parent
        [
            hint2type(hints[arg]) if arg in hints else None
            for arg in co.co_varnames[:co.co_argcount]
        ]
        +
        # then kwonly, going by the order in the child
        [
            hint2type(hints[arg.arg_name]) if arg.arg_name in hints else None
            for arg in child_args[co.co_argcount:]
        ]
    )


def get_typeshed_arg_types(
    parents_func: FunctionDescriptor|FunctionType,
    child_args: tuple[ArgInfo, ...]
) -> list[TypeInfo|None] | None:
    """Returns typeshed type annotations for a parent's method's arguments."""

    def find_def(tree: ast.AST, qualified_name: str) -> list[ast.FunctionDef|ast.AsyncFunctionDef]:
        parts = qualified_name.split('.')
        results = []

        def visit(node, path):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                full_name = '.'.join(path + [node.name])
                if full_name == qualified_name:
                    results.append(node)
            elif isinstance(node, ast.ClassDef):
                new_path = path + [node.name]
                for body_item in node.body:
                    visit(body_item, new_path)
            elif isinstance(node, ast.Module):
                for body_item in node.body:
                    visit(body_item, path)

        visit(tree, [])
        return results

    if stub_ast := ts.get_stub_ast(parents_func.__module__):    # FIXME replace __main__?
        if defs := find_def(stub_ast, parents_func.__qualname__):
            #print(ast.dump(defs[0], indent=4))

            # FIXME use eval() in the context of the module and hint2type so
            # as not to have an unqualified string for a "type"

            # first the positional, looking up by their names given in the parent
            pos_args = [
                TypeInfo('', ast.unparse(a.annotation)) if a.annotation else None
                for a in (defs[0].args.posonlyargs + defs[0].args.args)
                if isinstance(a, ast.arg)
            ]

            # then kwonly, going by the order in the child
            kw_args = [
                TypeInfo('', ast.unparse(a.annotation)) if a.annotation else None
                for child_arg_name in child_args[len(pos_args):]
                for a in defs[0].args.kwonlyargs
                if isinstance(a, ast.arg)
                if a.arg == child_arg_name
            ]

            return pos_args + kw_args

    return None


def resolve_mock(ti: TypeInfo, adjuster: AdjustTypeNamesT|None) -> TypeInfo|None:
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


@dataclass
class Observations:
    # Visited functions' and information about them
    functions_visited: dict[CodeId, FuncInfo] = field(default_factory=dict)

    # Started, but not (yet) completed traces
    pending_traces: dict[CodeId, dict[FrameId, PendingCallTrace]] = field(default_factory=lambda: defaultdict(dict))

    # Completed traces
    traces: dict[CodeId, Counter[CallTrace]] = field(default_factory=dict)

    # Mapping of sources to their module names
    # TODO handle cases where modules are loaded more than once, e.g. through pytest
    source_to_module_name: dict[str, str|None] = field(default_factory=dict)

    # target __main__ module globals
    main_globals: dict[str, Any]|None = None


    def record_module(
        self,
        code: CodeType,
        frame: FrameType
    ) -> None:
        if code.co_filename not in self.source_to_module_name:
            if (modname := frame.f_globals.get('__name__', None)):
                if modname == "__main__":
                    modname = get_main_module_fqn()
            else:
                modname = source_to_module_fqn(Path(code.co_filename))

            self.source_to_module_name[code.co_filename] = modname


    def record_function(
        self,
        code: CodeType,
        frame: FrameType,
        args: inspect.ArgInfo,
        overrides: FunctionType|FunctionDescriptor|None
    ) -> None:
        """Records that a function was visited, along with some details about it."""

        code_id = id(code)
        if code_id not in self.functions_visited:
            arg_names = (
                *(a for a in args.args),
                *((args.varargs,) if args.varargs else ()),
                *((args.keywords,) if args.keywords else ())
            )

            defaults = get_defaults(code, frame)

            self.functions_visited[code_id] = FuncInfo(
                FuncId(
                    Filename(code.co_filename),
                    code.co_firstlineno,
                    FunctionName(code.co_qualname),
                ),
                tuple(
                    ArgInfo(ArgumentName(name), defaults.get(name))
                    for name in arg_names
                ),
                ArgumentName(args.varargs) if args.varargs else None,
                ArgumentName(args.keywords) if args.keywords else None,
                overrides
            )


    @staticmethod
    def _get_arg_types(arg_info: inspect.ArgInfo) -> tuple[TypeInfo, ...]:
        """Computes the types of the given arguments."""
        return (
            *(get_value_type(arg_info.locals[arg_name]) for arg_name in arg_info.args),
            *(
                (TypeInfo.from_set({
                    get_value_type(val) for val in arg_info.locals[arg_info.varargs]
                }),)
                if arg_info.varargs else ()
            ),
            *(
                (TypeInfo.from_set({
                    get_value_type(val) for val in arg_info.locals[arg_info.keywords].values()
                }),)
                if arg_info.keywords else ()
            )
        )


    def record_start(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo
    ) -> None:
        """Records a function start."""

        # print(f"record_start {code.co_qualname} {arg_types}")
        self_type, self_replacement, overrides = get_self_type(code, arg_info)
        obs.record_function(code, frame, arg_info, overrides)
        obs.record_module(code, frame)

        self.pending_traces[id(code)][id(frame)] = PendingCallTrace(
            arg_info=arg_info,
            args=self._get_arg_types(arg_info),
            is_async=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE)),
            is_generator=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)),
            self_type=self_type, self_replacement=self_replacement
        )


    def record_yield(self, code: CodeType, frame_id: FrameId, yield_value: Any) -> None:
        """Records a yield."""

        # print(f"record_yield {code.co_qualname}")
        if (per_frame := self.pending_traces.get(id(code))) and (tr := per_frame.get(frame_id)):
            tr.yields.add(get_value_type(yield_value))


    def record_send(self, code: CodeType, frame_id: FrameId, send_value: Any) -> None:
        """Records a send."""

        # print(f"record_send {code.co_qualname}")
        if (per_frame := self.pending_traces.get(id(code))) and (tr := per_frame.get(frame_id)):
            tr.sends.add(get_value_type(send_value))


    def _record_return_type(self, tr: PendingCallTrace, code_id: CodeId, ret_type: Any) -> None:
        """Records a pending call trace's return type, finishing the trace."""
        assert tr is not None

        tr.returns = (
            ret_type if ret_type is not None
            else (
                # Generators may still be running, or exit with a GeneratorExit exception; we still
                # want them marked as returning None, so they can be simplified to Iterator
                NoneTypeInfo if tr.is_generator else TypeInfo.from_type(typing.NoReturn)
            )
        )

        if code_id not in self.traces:
            self.traces[code_id] = Counter()
        self.traces[code_id].update((tr.process(),))

        # Resample arguments in case they change during execution (e.g., containers)
        tr.args = self._get_arg_types(tr.arg_info)
        self.traces[code_id].update((tr.process(),))


    def record_return(self, code: CodeType, frame: FrameType, return_value: Any) -> bool:
        """Records a return."""

        # print(f"record_return {code.co_qualname}")
        code_id = id(code)
        frame_id = id(frame)
        if (per_frame := self.pending_traces.get(code_id)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code_id, get_value_type(return_value))
            del per_frame[frame_id]
            return True # found it

        return False


    def record_no_return(self, code: CodeType, frame: FrameType) -> bool:
        """Records the lack of a return (e.g., because an exception was raised)."""

        # print(f"record_no_return {code.co_qualname}")
        code_id = id(code)
        frame_id = id(frame)
        if (per_frame := self.pending_traces.get(code_id)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code_id, None)
            del per_frame[frame_id]
            return True # found it

        return False


    def _transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this class."""

        for code_id, trace_counter in self.traces.items():
            for trace, count in list(trace_counter.items()):
                trace_prime = tuple(tr.visit(t) for t in trace)
                # Use identity rather than ==, as only non-essential attributes may have changed
                if any(old is not new for old, new in zip(trace, trace_prime)):
                    if logger.level == logging.DEBUG:
                        func_info = self.functions_visited.get(code_id, None)
                        logger.debug(
                            type(tr).__name__ + " " +
                            (func_info.func_id.func_name if func_info else "?") +
                            str(tuple(str(t) for t in trace)) +
                            " -> " +
                            str(tuple(str(t) for t in trace_prime))
                        )
                    del trace_counter[trace]
                    trace_counter[trace_prime] = count


    def collect_annotations(self: Self) -> dict[FuncId, FuncAnnotation]:
        """Collects function type annotations from the observed types."""

        # Finish traces for any generators that may be still running
        for code_id, per_frame in self.pending_traces.items():
            for tr in per_frame.values():
                if tr.is_generator:
                    self._record_return_type(tr, code_id, None)


        def most_common_traces(code_id: CodeId) -> list[CallTrace]:
            """Returns the top X% most common call traces, turning type checking into anomaly detection."""
            counter = self.traces[code_id]

            threshold = sum(counter.values()) * options.use_top_pct / 100
            cumulative = 0

            traces = list()
            for trace, count in self.traces[code_id].most_common():
                if cumulative >= threshold:
                    break
                cumulative += count
                traces.append(trace)

            return traces


        def mk_annotation(code_id: CodeId) -> FuncAnnotation|None:
            func_info = self.functions_visited[code_id]
            traces = most_common_traces(code_id)

            parents_arg_types = None
            if func_info.overrides:
                parents_func = func_info.overrides

                if (
                    options.ignore_annotations
                    or not (
                        (parents_arg_types := get_inline_arg_types(parents_func, func_info.args))
                        or (parents_arg_types := get_typeshed_arg_types(parents_func, func_info.args))
                    )
                ):
                    parent_code_id = id(parents_func.__code__) if hasattr(parents_func, "__code__") else None
                    if (
                        parent_code_id
                        and parent_code_id in self.traces
                        and (ann := mk_annotation(parent_code_id))
                    ):
                        parents_arg_types = [arg[1] for arg in ann.args]

            if (signature := generalize(traces)) is None:
                logger.info(f"Unable to generalize {func_info.func_id}: inconsistent traces.\n" +
                            f"{[tuple(str(t) for t in s) for s in traces]}")
                return None

            ann = FuncAnnotation(
                args=[
                    (
                        arg.arg_name,
                        merged_types({
                            signature[i],
                            *((arg.default,) if arg.default is not None else ()),
                            # by building sets with the parent's types, we prevent arg. type narrowing
                            *((parents_arg_types[i],) if (
                                  parents_arg_types
                                  and len(parents_arg_types) == len(func_info.args)
                                  and parents_arg_types[i] is not None
                                ) else ()
                             )
                        })
                    )
                    for i, arg in enumerate(func_info.args)
                ],
                retval=signature[-1]
            )

            if logger.level == logging.DEBUG:
                trace_counter = self.traces[code_id]
                for trace, count in list(trace_counter.items()):
                    logger.debug(
                        "trace " + func_info.func_id.func_name +
                        str(tuple(str(t) for t in trace)) +
                        f" {count}x"
                    )
                logger.debug(
                    "ann   " + func_info.func_id.func_name +
                    str((*(str(arg[1]) for arg in ann.args), str(ann.retval)))
                )

            return ann

        type_name_adjuster = None
        if options.adjust_type_names:
            type_name_adjuster = AdjustTypeNamesT(self.main_globals)
            self._transform_types(type_name_adjuster)

        class NonSelfCloningT(TypeInfo.Transformer):
            """Clones the given TypeInfo tree, clearing all 'is_self' flags,
               as the type information may not be equivalent to typing.Self in the new context.
            """
            def visit(vself, node: TypeInfo) -> TypeInfo:
                return super().visit(node.replace(is_self=False))

        class CallableT(TypeInfo.Transformer):
            """Updates Callable/Generator/Coroutine type declarations based on observations."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                # if 'args' is there, the function is already annotated
                if node.code_id and (options.ignore_annotations or not node.args) and node.code_id in self.traces:
                    # TODO we only need the retval, can we avoid computing the entire annotation?
                    if (ann := mk_annotation(node.code_id)):
                        func_info = self.functions_visited[node.code_id]
                        # Clone (rather than link to) types from Callable, Generator, etc.,
                        # clearing is_self, as these types may be later replaced with typing.Self.
                        if node.type_obj is abc.Callable:
                            node = node.replace(args=(
                                TypeInfo.list([
                                    NonSelfCloningT().visit(a[1]) for a in ann.args[int(node.is_bound):]
                                ])
                                if not (func_info.varargs or func_info.kwargs) else
                                ...,
                                NonSelfCloningT().visit(ann.retval)
                            ))
                        elif node.type_obj in (abc.Generator, abc.AsyncGenerator):
                            node = NonSelfCloningT().visit(ann.retval)
                        elif node.type_obj is abc.Coroutine:
                            node = node.replace(args=(
                                NoneTypeInfo,
                                NoneTypeInfo,
                                NonSelfCloningT().visit(ann.retval)
                            ))

                return node

        self._transform_types(CallableT())

        class PostponedIteratorArgsT(TypeInfo.Transformer):
            """Replaces a postponed iterator argument evaluation marker with
               its evaluation (performed by CallableT).
            """
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                if node.type_obj is PostponedIteratorArg:
                    assert isinstance(node.args[0], TypeInfo)
                    source = node.args[0]
                    if source.args:
                        if source.type_obj is abc.Callable:
                            assert isinstance(source.args[1], TypeInfo)
                            return source.args[1]   # Callable return value
                        else:
                            assert isinstance(source.args[0], TypeInfo)
                            return source.args[0]   # Generator/Iterator yield value
                    return UnknownTypeInfo

                return node

        self._transform_types(PostponedIteratorArgsT())

        class SelfT(TypeInfo.Transformer):
            """Renames types to typing.Self according to is_self."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.is_self:
                    return TypeInfo("typing", "Self")

                return super().visit(node)

        if options.use_typing_self:
            self._transform_types(SelfT())

        class NeverSayNeverT(TypeInfo.Transformer):
            """Removes uses of typing.Never, replacing them with typing.Any"""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.type_obj is typing.Never:
                    return AnyTypeInfo

                return super().visit(node)

        class NoReturnIsNowNeverT(TypeInfo.Transformer):
            """Converts typing.NoReturn to typing.Never,
               which is the more modern way to type a 'no return'"""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.type_obj is typing.NoReturn:
                    return TypeInfo.from_type(typing.Never) 

                return super().visit(node)

        if not options.use_typing_never:
            self._transform_types(NeverSayNeverT())
        else:
            self._transform_types(NoReturnIsNowNeverT())

        class ResolveMocksT(TypeInfo.Transformer):
            """Resolves apparent test mock types to non-test ones."""
            # TODO make mock resolution context sensitive, leaving test-only
            # objects unresolved within test code?
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)
                if (resolved := resolve_mock(node, type_name_adjuster)):
                    return resolved
                return node

        if options.resolve_mocks:
            self._transform_types(ResolveMocksT())

        class ExcludeTestTypesT(TypeInfo.Transformer):
            """Removes test types."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if is_test_module(node.module):
                    return AnyTypeInfo

                return super().visit(node)

        if options.exclude_test_types:
            self._transform_types(ExcludeTestTypesT())

        class TypingUnionT(TypeInfo.Transformer):
            """Replaces types.UnionType with typing.Union and typing.Optional."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                # Typevar nodes may be UnionType; there's no need to replace them, and
                # replacing them would prevent RightTyper from annotating as typevars.
                if node.type_obj is UnionType and not node.is_typevar():
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
            def __init__(vself, limit: int):
                vself._limit = limit
                vself._level = -1
                vself._maxLevel = -1

            def visit(vself, node: TypeInfo) -> TypeInfo:
                # Don't count lists (such as the arguments in a Callable) as a level,
                # as it's not really a new type.
                if node.is_list():
                    return super().visit(node)

                try:
                    vself._level += 1
                    vself._maxLevel = max(vself._maxLevel, vself._level)

                    t = super().visit(node)

                    if vself._maxLevel > vself._limit:
                        # for containers, we can simply delete arguments (they default to Any)
                        if (
                            (type(t.type_obj) is type and issubclass(t.type_obj, abc.Container))
                            or t.type_obj is abc.Callable
                        ):
                            vself._maxLevel = vself._level
                            return t.replace(args=())

                    return t
                finally:
                    vself._level -= 1

        class ClearTypeObjTransformer(TypeInfo.Transformer):
            """Clears type_obj on all TypeInfo: annotations are pickled by 'multiprocessing',
               but many type objects (such as local ones, or from __main__) aren't pickleable.
            """
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.type_obj is not None:
                    node = node.replace(type_obj=None)
                return super().visit(node)

        finalizers: list[TypeInfo.Transformer] = []

        if options.type_depth_limit is not None:
            finalizers.append(DepthLimitT(options.type_depth_limit))

        if options.use_typing_union:
            finalizers.append(TypingUnionT())

        finalizers.append(ClearTypeObjTransformer())

        def finalize(t: TypeInfo) -> TypeInfo:
            for f in finalizers:
                t = f.visit(t)
            return t

        return {
            self.functions_visited[code_id].func_id: FuncAnnotation(
                args=[(arg[0], finalize(arg[1])) for arg in annotation.args],
                retval=finalize(annotation.retval)
            )
            for code_id in self.traces
            if (annotation := mk_annotation(code_id)) is not None
        }


obs = Observations()

instrumentation_counter = AtomicCounter()


def is_instrumentation(f):
    """Decorator that marks a function as being instrumentation."""
    def wrapper(*args, **kwargs):
        try:
            instrumentation_counter.inc()
            return f(*args, **kwargs)
        finally:
            instrumentation_counter.dec()

    return wrapper


@is_instrumentation
def send_handler(code: CodeType, frame_id: FrameId, arg0: Any) -> None:
    obs.record_send(
        code,
        frame_id, 
        arg0
    )


def wrap_send(obj: Any) -> Any:
    if (
        (self := getattr(obj, "__self__", None)) and
        isinstance(self, (GeneratorType, AsyncGeneratorType))
    ):
        if isinstance(self, GeneratorType):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                # generator.send takes exactly one argument
                send_handler(self.gi_code, FrameId(id(self.gi_frame)), args[0])
                return obj(*args, **kwargs)

            return wrapper
        else:
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                # generator.asend takes exactly one argument
                send_handler(self.ag_code, FrameId(id(self.ag_frame)), args[0])
                return obj(*args, **kwargs)

            return wrapper

    return obj


@is_instrumentation
def start_handler(code: CodeType, offset: int) -> Any:
    """
    Process the function entry point, perform monitoring related operations,
    and manage the profiling of function execution.
    """
    if should_skip_function(code) or id(code) in disabled_code:
        return sys.monitoring.DISABLE

    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    if frame:
        obs.record_start(code, frame, inspect.getargvalues(frame))
        del frame

    return None


@is_instrumentation
def call_handler(
    code: CodeType,
    instruction_offset: int,
    callable: object,
    arg0: object,
) -> Any:
    # If we are calling a function, activate its start, return, and yield handlers.
    if isinstance(callable, FunctionType) and isinstance(getattr(callable, "__code__", None), CodeType):
        if not should_skip_function(code):
            sys.monitoring.set_local_events(
                TOOL_ID,
                callable.__code__,
                sys.monitoring.events.PY_START
                | sys.monitoring.events.PY_RETURN
                | sys.monitoring.events.PY_YIELD
            )

    return sys.monitoring.DISABLE


@is_instrumentation
def yield_handler(
    code: CodeType,
    instruction_offset: int,
    yield_value: Any,
) -> Any:
    """
    Processes a yield event for a function.

    Args:
    code (CodeType): code object of the function.
    instruction_offset (int): position of the current instruction.
    yield_value (Any): return value of the function.
    """
    if should_skip_function(code) or id(code) in disabled_code:
        return sys.monitoring.DISABLE

    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    if frame:
        obs.record_yield(code, id(frame), yield_value)
        del frame

    return None


@is_instrumentation
def return_handler(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
) -> Any:
    """
    Processes a return event for a function.

    Args:
    code (CodeType): code object of the function.
    instruction_offset (int): position of the current instruction.
    return_value (Any): return value of the function.
    """
    if should_skip_function(code) or id(code) in disabled_code:
        return sys.monitoring.DISABLE

    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    found = frame and obs.record_return(code, frame, return_value)
    del frame

    if (
        found
        and options.sampling
        and not (
            (no_sampling_for := options.no_sampling_for_re)
            and no_sampling_for.search(code.co_qualname)
        )
    ):
        disabled_code.add(id(code))
        obs.pending_traces[id(code)].clear()
        return sys.monitoring.DISABLE

    return None


@is_instrumentation
def unwind_handler(
    code: CodeType,
    instruction_offset: int,
    exception: BaseException,
) -> Any:

    if should_skip_function(code):
        return None # PY_UNWIND can't be disabled

    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    found = frame and obs.record_no_return(code, frame)
    del frame

    if (
        found
        and options.sampling
        and not (
            (no_sampling_for := options.no_sampling_for_re)
            and no_sampling_for.search(code.co_qualname)
        )
    ):
        disabled_code.add(id(code))
        obs.pending_traces[id(code)].clear()

    return None # PY_UNWIND can't be disabled


def get_self_type(code, args) -> tuple[TypeInfo|None, TypeInfo|None, FunctionType|FunctionDescriptor|None]:
    if args.args:
        first_arg = args.locals[args.args[0]]

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
                    # wrapper_descriptor and possibly other native objects may lack __module__
                    f if hasattr(f, "__module__")
                    else FunctionDescriptor(ancestor.__module__, f.__qualname__)
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
            if param.default != inspect._empty
        }

    return {}


sample_count_instrumentation = 0
sample_count_total = 0
overhead: float|None = None
samples_instrumentation = []
samples_total = []
instrumentation_overhead = []
instrumentation_restarted = []
disabled_code: set[CodeId] = set()

def exp_smooth(value: float, previous: float|None) -> float:
    """Exponentially smooths a value."""
    if previous is None: return value

    alpha = 0.4
    return alpha * value + (1-alpha) * previous


def self_profile() -> None:
    """
    Measures the instrumentation overhead, restarting event delivery
    if it lies below the target overhead.
    """
    global sample_count_instrumentation, sample_count_total, overhead
    sample_count_total += 1
    if instrumentation_counter.count() > 0:
        sample_count_instrumentation += 1

    # Only calculate overhead every so often, so as to allow the currently
    # enabled events to be triggered.  Doing it every time makes it jumpy
    if (sample_count_total % 50) == 0:
        interval = float(sample_count_instrumentation) / sample_count_total
        overhead = exp_smooth(interval, overhead)

        if (restart := (overhead <= options.target_overhead / 100.0)):
            # Instrumentation overhead is low enough: restart instrumentation.
            disabled_code.clear()
            sys.monitoring.restart_events()

        if options.save_profiling is not None:
            samples_instrumentation.append(sample_count_instrumentation)
            samples_total.append(sample_count_total)
            instrumentation_overhead.append(overhead)
            instrumentation_restarted.append(restart)

        sample_count_instrumentation = sample_count_total = 0


def execute_script_or_module(
    script: str,
    is_module: bool,
    args: list[str],
) -> None:
    """Executes the script or module, returning the __main__ module's globals."""

    try:
        sys.argv = [script, *args]
        if is_module:
            with loader.ImportManager(replace_dict=options.replace_dict):
                obs.main_globals = runpy.run_module(
                    script,
                    run_name="__main__",
                    alter_sys=True,
                )
        else:
            with loader.ImportManager(replace_dict=options.replace_dict):
                obs.main_globals = runpy.run_path(script, run_name="__main__")

    except SystemExit as e:
        if e.code not in (None, 0):
            raise

    # TODO: save main_globals somehow upon exception


def output_signatures(
    sig_changes: list[SignatureChanges],
    file: TextIO = sys.stdout,
) -> None:
    import difflib

    for filename, changes in sorted(sig_changes):
        if not changes:
            continue

        print(
            f"{filename}:\n{'=' * (len(filename) + 1)}\n",
            file=file,
        )

        for funcname, old, new in sorted(changes):
            print(f"{funcname}\n", file=file)

            # show signature diff
            diffs = difflib.ndiff(
                (old + "\n").splitlines(True),
                (new + "\n").splitlines(True),
            )
            print("".join(diffs), file=file)


def process_collected(collected: dict[str, Any]):
    sig_changes: list[SignatureChanges] = process_files(
        collected['files'],
        collected['type_annotations']
    )

    if options.json_output:
        data: dict[str, Any] = {
            'meta': {
                'software': TOOL_NAME,
                'version': importlib.metadata.version(TOOL_NAME),
                'timestamp': datetime.datetime.now().isoformat(),
            },
            'files': dict()
        }

        file2module = {file: module for file, module in collected['files']}
        file_func2sigs = {
            (file, funcname): (old_sig, new_sig)
            for file, changes in sig_changes
            for funcname, old_sig, new_sig in changes
        }

        for funcid in sorted(collected['type_annotations']):
            if funcid.file_name not in data['files']:
                entry: dict[str, Any]
                entry = data['files'][funcid.file_name] = {
                    'module': file2module.get(funcid.file_name),
                    'functions': dict()
                }

            if funcid.func_name in entry['functions']:
                continue  # TODO handle multiple first_code_line

            ann = collected['type_annotations'][funcid]
            entry['functions'][funcid.func_name] = {
                'args': {a[0]: str(a[1]) for a in ann.args},
                'retval': str(ann.retval)
            }

            if changes := file_func2sigs.get((funcid.file_name, funcid.func_name)):
                entry['functions'][funcid.func_name]['old_sig'] = changes[0]
                entry['functions'][funcid.func_name]['new_sig'] = changes[1]

        with open(f"{TOOL_NAME}.json", "w") as f:
            json.dump(data, f, indent=2)

    else:
        with open(f"{TOOL_NAME}.out", "w+") as f:
            output_signatures(sig_changes, f)


def process_file_wrapper(args) -> SignatureChanges:
    return process_file(*args)


def process_files(
    files: list[list[str]],
    type_annotations: dict[FuncId, FuncAnnotation],
) -> list[SignatureChanges]:
    if not files:
        return []

    args_gen = (
        (
            file[0],    # path
            file[1],    # module_name
            type_annotations,
            options.output_files,
            options.generate_stubs,
            options.overwrite,
            options.ignore_annotations,
            options.only_update_annotations,
            options.inline_generics
        )
        for file in files
    )

    if options.use_multiprocessing:
        def mp_map(fn, args_gen):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for fut in concurrent.futures.as_completed([
                    executor.submit(process_file_wrapper, args)
                    for args in args_gen
                ]):
                    yield fut.result()

        results = mp_map(process_file_wrapper, args_gen)
    else:
        results = map(process_file_wrapper, args_gen)

    # 'rich' is unusable right after running its test suite,
    # so reload it just in case we just did that.
    if 'rich' in sys.modules:
        importlib.reload(sys.modules['rich'])
        importlib.reload(sys.modules['rich.progress'])

    import rich.progress
    from rich.table import Column

    sig_changes = []

    with rich.progress.Progress(
        rich.progress.BarColumn(table_column=Column(ratio=1)),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        transient=True,
        expand=True,
        auto_refresh=False,
    ) as progress:
        task1 = progress.add_task(description="", total=len(files))

        for result in results:
            sig_changes.append(result)
            progress.update(task1, advance=1)
            progress.refresh()

    return sig_changes


def validate_module(ctx, param, value):
    if not value or importlib.util.find_spec(value):
        return value
    raise click.BadParameter("not a valid module.")


def validate_module_names(ctx, param, value):
    for m in value:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$', m):
            raise click.BadParameter(f""""{m}" is not a valid module.""")

    return value


def parse_none_or_ge_zero(value) -> int|None:
    if value.lower() == "none":
        return None
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise click.BadParameter("must be ≥ 0 or 'none'")
        return ivalue
    except ValueError:
        raise click.BadParameter("must be an integer ≥ 0 or 'none'")


def validate_regexes(ctx, param, value):
    try:
        if value:
            for v in value:
                re.compile(v)
        return value
    except re.error as e:
        raise click.BadParameter(str(e))


@click.group(
    context_settings={
        "show_default": True
    }
)
@click.option(
    # just for backwards compatibility
    "--debug",
    is_flag=True,
    hidden=True,
)
@click.version_option(
    version=importlib.metadata.version(TOOL_NAME),
    prog_name=TOOL_NAME,
)
def cli(debug: bool):
    if debug:
        logger.setLevel(logging.DEBUG)


@cli.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
@click.argument(
    "script",
    required=False,
)
@click.option(
    "-m",
    "--module",
    help="Run the given module instead of a script.",
    callback=validate_module
)
@click.option(
    "--all-files",
    is_flag=True,
    help="Process any files encountered, including libraries (except for those specified in --include-files)",
)
@click.option(
    "--include-files",
    metavar="REGEX",
    type=str,
    multiple=True,
    callback=validate_regexes,
    help="Process only files matching the given regular expression. Can be passed multiple times.",
)
@click.option(
    "--exclude-test-files/--no-exclude-test-files",
    default=options.exclude_test_files,
    help="Automatically exclude test modules from typing.",
)
@click.option(
    "--include-functions",
    metavar="REGEX",
    type=str,
    multiple=True,
    callback=validate_regexes,
    help="Only annotate functions matching the given regular expression. Can be passed multiple times.",
)
@click.option(
    "--infer-shapes",
    is_flag=True,
    default=False,
    show_default=True,
    help="Produce tensor shape annotations (compatible with jaxtyping).",
)
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False),
    help="Process only files under the given directory.  If omitted, the script's directory (or, for -m, the current directory) is used.",
)
@click.option(
    "--overwrite/--no-overwrite",
    help="""Overwrite ".py" files with type information. If disabled, ".py.typed" files are written instead. The original files are saved as ".py.bak".""",
    default=options.overwrite,
)
@click.option(
    "--output-files/--no-output-files",
    help=f"Output annotated files (possibly overwriting, if specified).  If disabled, the annotations are only written to {TOOL_NAME}.out.",
    default=options.output_files,
)
@click.option(
    "--ignore-annotations",
    is_flag=True,
    help="Ignore existing annotations and overwrite with type information.",
    default=False,
)
@click.option(
    "--only-update-annotations",
    is_flag=True,
    default=False,
    help="Overwrite existing annotations but never add new ones.",
)
@click.option(
    "--generate-stubs",
    is_flag=True,
    help="Generate stub files (.pyi).",
    default=False,
)
@click.option(
    "--json-output",
    default=options.json_output,
    is_flag=True,
    help=f"Output inferences in JSON, instead of writing {TOOL_NAME}.out."
)
@click.option(
    "--target-overhead",
    type=float,
    default=options.target_overhead,
    help="Target overhead, as a percentage (e.g., 5).",
)
@click.option(
    "--use-multiprocessing/--no-use-multiprocessing",
    default=True,
    help="Whether to use multiprocessing.",
)
@click.option(
    "--sampling/--no-sampling",
    default=options.sampling,
    help=f"Whether to sample calls or to use every one.",
)
@click.option(
    "--no-sampling-for",
    metavar="REGEX",
    type=str,
    multiple=True,
    callback=validate_regexes,
    default=options.no_sampling_for,
    help=f"Rather than sample, record every invocation of any functions matching the given regular expression. Can be passed multiple times.",
)
@click.option(
    "--signal-wakeup/--thread-wakeup",
    default=not platform.system() == "Windows",
    hidden=True,
    help="Whether to use signal-based wakeups or thread-based wakeups."
)
@click.option(
    "--replace-dict/--no-replace-dict",
    is_flag=True,
    help="Whether to replace 'dict' to enable efficient, statistically correct samples."
)
@click.option(
    "--container-sample-limit",
    default="none",
    callback=lambda ctx, param, value: parse_none_or_ge_zero(value),
    show_default=True,
    metavar="[INTEGER|none]",
    help="Number of container elements to sample; 'none' to disable.",
)
@click.option(
    "--type-depth-limit",
    default="none",
    callback=lambda ctx, param, value: parse_none_or_ge_zero(value),
    show_default=True,
    metavar="[INTEGER|none]",
    help="Maximum depth (types within types) for generic types; 'none' to disable.",
)
@click.option(
    "--python-version",
    type=click.Choice(["3.9", "3.10", "3.11", "3.12", "3.13"]),
    default="3.12",
    callback=lambda ctx, param, value: tuple(int(n) for n in value.split('.')),
    help="Python version for which to emit annotations.",
)
@click.option(
    "--use-top-pct",
    type=click.IntRange(1, 100),
    default=options.use_top_pct,
    metavar="PCT",
    help="Only use the PCT% most common call traces.",
)
@click.option(
    "--only-collect",
    default=False,
    is_flag=True,
    help=f"Rather than immediately process collect data, save it to {PKL_FILE_NAME}." +\
          " You can later process using RightTyper's \"process\" command."
)
@click.option(
    "--save-profiling",
    type=str,
    metavar="NAME",
    hidden=True,
    help=f"""Save record of self-profiling results in "{TOOL_NAME}-profiling.json", under the given name."""
)
@click.option(
    "--resolve-mocks/--no-resolve-mocks",
    is_flag=True,
    default=options.resolve_mocks,
    help="Whether to attempt to resolve test types, such as mocks, to non-test types."
)
@click.option(
    "--exclude-test-types/--no-exclude-test-types",
    is_flag=True,
    default=options.exclude_test_types,
    help="""Whether to exclude or replace with "typing.Any" types defined in test modules."""
)
@click.option(
    "--test-modules",
    multiple=True,
    default=options.test_modules,
    callback=validate_module_names,
    metavar="MODULE",
    help="""Additional modules (besides those detected) whose types are subject to mock resolution or test type exclusion, if enabled. Matches submodules as well. Can be passed multiple times."""
)
@click.option(
    "--use-typing-never/--no-use-typing-never",
    default=True,
    help="""Whether to emit "typing.Never".""",
)
@click.option(
    "--adjust-type-names/--no-adjust-type-names",
    default=options.adjust_type_names,
    help="Whether to look for a canonical name for types, rather than use the module and name where they are defined.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Include diagnostic information in log file.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def run(
    script: str,
    module: str,
    args: list[str],
    all_files: bool,
    include_files: tuple[str, ...],
    exclude_test_files: bool,
    include_functions: tuple[str, ...],
    overwrite: bool,
    output_files: bool,
    ignore_annotations: bool,
    only_update_annotations: bool,
    generate_stubs: bool,
    json_output: bool,
    infer_shapes: bool,
    root: str,
    target_overhead: float,
    use_multiprocessing: bool,
    sampling: bool,
    no_sampling_for: tuple[str, ...],
    signal_wakeup: bool,
    replace_dict: bool,
    container_sample_limit: int,
    python_version: tuple[int, ...],
    use_top_pct: int,
    only_collect: bool,
    save_profiling: str,
    type_depth_limit: int|None,
    resolve_mocks: bool,
    exclude_test_types: bool,
    test_modules: tuple[str, ...],
    use_typing_never: bool,
    adjust_type_names: bool,
    debug: bool
) -> None:
    """Runs a given script or module, collecting type information."""

    start_time = time.perf_counter()

    logger.info(f"Starting: {subprocess.list2cmdline(sys.orig_argv)}")
    if debug:
        logger.setLevel(logging.DEBUG)

    if module:
        args = [*((script,) if script else ()), *args]  # script, if any, is really the 1st module arg
        script = module
    elif script:
        if not os.path.isfile(script):
            raise click.UsageError(f"\"{script}\" is not a file.")
    else:
        raise click.UsageError("Either -m/--module must be provided, or a script be passed.")

    if ignore_annotations and only_update_annotations:
        raise click.UsageError("Options --ignore-annotations and --only-update-annotations are mutually exclusive.")

    if infer_shapes:
        # Check for required packages for shape inference
        found_package = defaultdict(bool)
        packages = ["jaxtyping"]
        all_packages_found = True
        for package in packages:
            found_package[package] = (
                importlib.util.find_spec(package) is not None
            )
            all_packages_found &= found_package[package]
        if not all_packages_found:
            print("The following package(s) need to be installed:")
            for package in packages:
                if not found_package[package]:
                    print(f" * {package}")
            sys.exit(1)

    if root:
        options.script_dir = os.path.realpath(root)
    elif module:
        options.script_dir = os.getcwd()
    else:
        options.script_dir = os.path.dirname(os.path.realpath(script))

    options.include_files = include_files
    options.include_all = all_files
    options.exclude_test_files = exclude_test_files
    options.include_functions = include_functions
    options.target_overhead = target_overhead
    options.infer_shapes = infer_shapes
    options.ignore_annotations = ignore_annotations
    options.only_update_annotations = only_update_annotations
    options.overwrite = overwrite
    options.output_files = output_files
    options.generate_stubs = generate_stubs
    options.json_output = json_output
    options.use_multiprocessing = use_multiprocessing
    options.sampling = sampling
    options.no_sampling_for = no_sampling_for
    options.replace_dict = replace_dict
    options.container_sample_limit = container_sample_limit
    options.use_typing_union = python_version < (3, 10)
    options.use_typing_self = python_version >= (3, 11)
    options.use_typing_never = python_version >= (3, 11) and use_typing_never
    options.inline_generics = python_version >= (3, 12)
    options.use_top_pct = use_top_pct
    options.type_depth_limit = type_depth_limit
    options.resolve_mocks = resolve_mocks
    options.exclude_test_types = exclude_test_types
    options.test_modules = test_modules
    options.adjust_type_names = adjust_type_names
    options.save_profiling = save_profiling

    alarm_cls = SignalAlarm if signal_wakeup else ThreadAlarm
    alarm = alarm_cls(self_profile, 0.01)

    pytest_plugins = os.environ.get("PYTEST_PLUGINS")
    pytest_plugins = (pytest_plugins + "," if pytest_plugins else "") + "righttyper.pytest"
    os.environ["PYTEST_PLUGINS"] = pytest_plugins

    register_monitoring_callbacks(
        start_handler,
        return_handler,
        yield_handler,
        call_handler,
        unwind_handler,
    )
    sys.monitoring.restart_events()
    alarm.start()

    try:
        execute_script_or_module(script, is_module=bool(module), args=args)
    finally:
        reset_monitoring()
        alarm.stop()

        file_names = set(
            t.func_id.file_name
            for t in obs.functions_visited.values()
            if not skip_this_file(t.func_id.file_name)
        )

        type_annotations = obs.collect_annotations()

        if logger.level == logging.DEBUG:
            for m in detected_test_modules:
                logger.debug(f"test module: {m}")

        collected = {
            'version': PKL_FILE_VERSION,
            'files': [[f, obs.source_to_module_name.get(f)] for f in file_names],
            'type_annotations': type_annotations,
            'options': options
        }

        logger.debug(f"observed {len(file_names)} file(s)")
        logger.debug(f"generated {len(type_annotations)} annotation(s)")

        if only_collect:
            with open(PKL_FILE_NAME, "wb") as pklf:
                pickle.dump(collected, pklf)

            print(f"Collected types saved to {PKL_FILE_NAME}.")
        else:
            process_collected(collected)

        end_time = time.perf_counter()
        logger.info(f"Finished in {end_time-start_time:.0f}s")

        if save_profiling:
            try:
                with open(f"{TOOL_NAME}-profiling.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = []

            data.append({
                    'name': save_profiling,
                    'command': subprocess.list2cmdline(sys.orig_argv),
                    'start_time': start_time,
                    'end_time': end_time,
                    'elapsed': end_time - start_time,
                    'overhead': instrumentation_overhead,
                    'restarted': instrumentation_restarted,
                    'samples_instrumentation': samples_instrumentation,
                    'samples_total': samples_total,
                }
            )

            with open(f"{TOOL_NAME}-profiling.json", "w") as f:
                json.dump(data, f, indent=2)


@cli.command()
def process():
    """Processes type information collected with the 'run' command."""

    try:
        with open(PKL_FILE_NAME, "rb") as f:
            pkl = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: No '{PKL_FILE_NAME}' found to process.")
        sys.exit(1)

    if pkl.get('version') != PKL_FILE_VERSION:
        print(f"Error: Unsupported {PKL_FILE_NAME} version: {pkl.get('version')}, expected {PKL_FILE_VERSION}")
        sys.exit(1)

    # Copy run options, but be careful not to replace instance, as there may be
    # multiple references to it (e.g., through "from .options import ...")
    global options
    for key, value in asdict(pkl['options']).items():
        setattr(options, key, value)

    process_collected(pkl)


@cli.command()
@click.option(
    "--type", 'cov_type',
    type=click.Choice(["by-directory", "by-file", "summary"]),
    default='summary',
    help="Select coverage type.",
)
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False),
)
def coverage(
    cov_type: str,
    path: Path
):
    """Computes annotation coverage."""

    from . import annotation_coverage as cov

    cache = cov.analyze_all_directories(str(path))

    if cov_type == "by-directory":
        cov.print_directory_summary(cache)
    elif cov_type == "by-file":
        cov.print_file_summary(cache)
    else:
        cov.print_annotation_summary()
