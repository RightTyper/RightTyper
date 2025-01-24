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


import collections.abc as abc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType, FunctionType, MethodType, GeneratorType, AsyncGeneratorType, UnionType
import typing
from typing import (
    Any,
    TextIO,
    Self
)

import click

# Disabled for now
# from righttyper import replace_dicts
from righttyper.righttyper_process import (
    process_file,
    SignatureChanges
)
from righttyper.righttyper_runtime import (
    find_function,
    get_override_contexts,
    unwrap,
    get_value_type,
    get_type_name,
    should_skip_function,
    hint2type,
    PostponedIteratorArg,
)
from righttyper.righttyper_tool import (
    register_monitoring_callbacks,
    reset_monitoring,
    setup_tool_id,
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
    FuncInstance,
    FunctionName,
    FunctionDescriptor,
    TypeInfo,
    NoneTypeInfo,
    AnyTypeInfo,
    Sample,
    UnknownTypeInfo
)
from righttyper.typeinfo import (
    merged_types,
    generalize,
)
from righttyper.righttyper_utils import (
    TOOL_ID,
    TOOL_NAME,
    debug_print_set_level,
    skip_this_file,
    get_main_module_fqn
)
import righttyper.loader as loader
from righttyper.righttyper_alarm import (
    SignalAlarm,
    ThreadAlarm,
)

@dataclass
class Options:
    script_dir: str = ""
    include_files_pattern: str = ""
    include_all: bool = False
    include_functions_pattern: tuple[str, ...] = tuple()
    target_overhead: float = 5.0
    infer_shapes: bool = False
    ignore_annotations: bool = False
    overwrite: bool = False
    output_files: bool = False
    generate_stubs: bool = False
    srcdir: str = ""
    use_multiprocessing: bool = True
    sampling: bool = True
    replace_dict: bool = False
    container_sample_limit: int = 1000
    use_typing_union: bool = False
    use_typing_self: bool = False
    use_typing_never: bool = False
    inline_generics: bool = False
    only_update_annotations: bool = False

options = Options()


logger = logging.getLogger("righttyper")


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
    except NameError:
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


@dataclass
class Observations:
    # Visited functions' and information about them
    functions_visited: dict[CodeId, FuncInfo] = field(default_factory=dict)

    # Started, but not (yet) completed samples
    pending_samples: dict[tuple[CodeId, FrameId], Sample] = field(default_factory=dict)

    # Completed samples
    samples: dict[CodeId, set[tuple[TypeInfo, ...]]] = field(default_factory=dict)


    def record_function(
        self,
        code: CodeType,
        args: inspect.ArgInfo,
        get_defaults: abc.Callable[[], dict[str, TypeInfo]],
        overrides: FunctionType|FunctionDescriptor|None,
        function_object: FuncInstance
    ) -> None:
        """Records that a function was visited, along with some details about it."""

        for override_code in get_override_contexts(function_object, code):
            code_id = CodeId(id(override_code))
            if code_id not in self.functions_visited:
                arg_names = (
                    *(a for a in args.args),
                    *((args.varargs,) if args.varargs else ()),
                    *((args.keywords,) if args.keywords else ())
                )

                defaults = get_defaults()

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


    def record_start(
        self,
        code: CodeType,
        frame_id: FrameId,
        arg_types: tuple[TypeInfo, ...],
        self_type: TypeInfo|None,
        self_replacement: TypeInfo|None,
        function_object: FuncInstance,
    ) -> None:
        """Records a function start."""

        # print(f"record_start {code.co_qualname} {arg_types}")
        self.pending_samples[(CodeId(id(code)), frame_id)] = Sample(
            arg_types,
            self_type=self_type,
            self_replacement=self_replacement,
            is_async=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE)),
            is_generator=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)),
            function_object=function_object,
        )


    def record_yield(self, code: CodeType, frame_id: FrameId, yield_type: TypeInfo) -> bool:
        """Records a yield."""

        # print(f"record_yield {code.co_qualname}")
        if (sample := self.pending_samples.get((CodeId(id(code)), frame_id))):
            sample.yields.add(yield_type)
            return True

        return False


    def record_send(self, code: CodeType, frame_id: FrameId, send_type: TypeInfo) -> bool:
        """Records a send."""

        # print(f"record_send {code.co_qualname}")
        if (sample := self.pending_samples.get((CodeId(id(code)), frame_id))):
            sample.sends.add(send_type)
            return True

        return False


    def record_return(self, code: CodeType, frame_id: FrameId, return_type: TypeInfo) -> bool:
        """Records a return."""

        # print(f"record_return {code.co_qualname}")

        code_id = CodeId(id(code))
        if (sample := self.pending_samples.get((code_id, frame_id))):
            sample.returns = return_type
            if code_id not in self.samples:
                self.samples[code_id] = set()
            # MyPy labels function_object as a MethodType instance since it think that this is a method access
            for overridden_code in get_override_contexts(sample.function_object, code): # type: ignore
                overridden_code_id = CodeId(id(overridden_code))
                self.samples[overridden_code_id].add(sample.process())
            del self.pending_samples[(code_id, frame_id)]
            return True

        return False


    def _transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this class."""

        for sample_set in self.samples.values():
            for s in list(sample_set):
                sprime = tuple(tr.visit(t) for t in s)
                # Use identity rather than ==, as only non-essential attributes may have changed
                if any(old is not new for old, new in zip(s, sprime)):
                    sample_set.remove(s)
                    sample_set.add(sprime)


    def collect_annotations(self: Self) -> dict[FuncId, FuncAnnotation]:
        """Collects function type annotations from the observed types."""

        # Finish samples for any generators that are still unfinished
        # TODO are there other cases we should handle?
        for (code_id, _), sample in self.pending_samples.items():
            if sample.yields:
                if code_id not in self.samples:
                    self.samples[code_id] = set()
                self.samples[code_id].add(sample.process())

        def mk_annotation(code_id: CodeId) -> FuncAnnotation|None:
            func_info = self.functions_visited[code_id]
            samples = self.samples[code_id]

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
                    parent_code_id = CodeId(id(parents_func.__code__)) if hasattr(parents_func, "__code__") else None
                    if (
                        parent_code_id
                        and parent_code_id in self.samples
                        and (ann := mk_annotation(parent_code_id))
                    ):
                        parents_arg_types = [arg[1] for arg in ann.args]

            if (signature := generalize(list(samples))) is None:
                logger.info(f"Unable to generalize {func_info.func_id}: inconsistent samples.\n" +
                            f"{[tuple(str(t) for t in s) for s in samples]}")
                return None

            return FuncAnnotation(
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
                if node.code_id and (options.ignore_annotations or not node.args) and node.code_id in self.samples:
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
                if node.qualname() == "typing.Never":
                    return TypeInfo("typing", "Any")

                return super().visit(node)

        if not options.use_typing_never:
            self._transform_types(NeverSayNeverT())

        class TypingUnionT(TypeInfo.Transformer):
            """Replaces types.UnionType with typing.Union and typing.Optional."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                if node.type_obj is UnionType:
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


        class ClearTypeObjTransformer(TypeInfo.Transformer):
            """Clears type_obj on all TypeInfo: annotations are pickled by 'multiprocessing',
               but many type objects (such as local ones, or from __main__) aren't pickleable.
            """
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.type_obj is not None:
                    node = node.replace(type_obj=None)
                return super().visit(node)


        clear = ClearTypeObjTransformer()

        if options.use_typing_union:
            tu = TypingUnionT()

            def finalize(t: TypeInfo) -> TypeInfo:
                return clear.visit(tu.visit(t))
        else:
            def finalize(t: TypeInfo) -> TypeInfo:
                return clear.visit(t)

        return {
            self.functions_visited[code_id].func_id: FuncAnnotation(
                args=[(arg[0], finalize(arg[1])) for arg in annotation.args],
                retval=finalize(annotation.retval)
            )
            for code_id in self.samples
            if (annotation := mk_annotation(code_id)) is not None
        }


obs = Observations()


def send_handler(code: CodeType, frame_id: FrameId, arg0: Any) -> None:
    obs.record_send(
        code,
        frame_id, 
        get_value_type(
            arg0,
            container_sample_limit=options.container_sample_limit,
            use_jaxtyping=options.infer_shapes
        )
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


def enter_handler(code: CodeType, offset: int) -> Any:
    """
    Process the function entry point, perform monitoring related operations,
    and manage the profiling of function execution.
    """
    if should_skip_function(
        code,
        options.script_dir,
        options.include_all,
        options.include_files_pattern,
        options.include_functions_pattern
    ):
        return sys.monitoring.DISABLE

    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    if frame:
        process_function_call(code, frame)
        del frame

    return sys.monitoring.DISABLE if options.sampling else None


def call_handler(
    code: CodeType,
    instruction_offset: int,
    callable: object,
    arg0: object,
) -> Any:
    # If we are calling a function, activate its start, return, and yield handlers.
    if isinstance(callable, FunctionType) and isinstance(getattr(callable, "__code__", None), CodeType):
        if not should_skip_function(
            code,
            options.script_dir,
            options.include_all,
            options.include_files_pattern,
            options.include_functions_pattern,
        ):
            sys.monitoring.set_local_events(
                TOOL_ID,
                callable.__code__,
                sys.monitoring.events.PY_START
                | sys.monitoring.events.PY_RETURN
                | sys.monitoring.events.PY_YIELD
            )

    return sys.monitoring.DISABLE


def yield_handler(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
) -> object:
    # We do the same thing for yields and exits.
    return process_yield_or_return(
        code,
        instruction_offset,
        return_value,
        sys.monitoring.events.PY_YIELD,
    )


def return_handler(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
) -> object:
    return process_yield_or_return(
        code,
        instruction_offset,
        return_value,
        sys.monitoring.events.PY_RETURN,
    )


def process_yield_or_return(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
    event_type: int,
) -> object:
    """
    Processes a yield or return event for a function.
    Function to gather statistics on a function call and determine
    whether it should be excluded from profiling, when the function exits.

    - If the function name is in the excluded list, it will disable the monitoring right away.
    - Otherwise, it calculates the execution time of the function, adds the type of the return value to a set for that function,
      and then disables the monitoring if appropriate.

    Args:
    code (CodeType): code object of the function.
    instruction_offset (int): position of the current instruction.
    return_value (Any): return value of the function.
    event_type (int): if this is a PY_RETURN (regular return) or a PY_YIELD (yield)

    Returns:
    int: indicator whether to continue the monitoring, always returns sys.monitoring.DISABLE in this function.
    """
    # Check if the function name is in the excluded list
    if should_skip_function(
        code,
        options.script_dir,
        options.include_all,
        options.include_files_pattern,
        options.include_functions_pattern
    ):
        return sys.monitoring.DISABLE

    found = False

    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    if frame:
        typeinfo = get_value_type(
            return_value,
            container_sample_limit=options.container_sample_limit,
            use_jaxtyping=options.infer_shapes
        )

        if event_type == sys.monitoring.events.PY_YIELD:
            found = obs.record_yield(code, FrameId(id(frame)), typeinfo)
        else:
            found = obs.record_return(code, FrameId(id(frame)), typeinfo)

        del frame

    # If the frame wasn't found, keep the event enabled, as this event may be from another
    # invocation whose start we missed.
    return sys.monitoring.DISABLE if (options.sampling and found) else None


def process_function_call(
    code: CodeType,
    frame: FrameType,
) -> None:

    def get_type(v: Any) -> TypeInfo:
        return get_value_type(
            v,
            container_sample_limit=options.container_sample_limit,
            use_jaxtyping=options.infer_shapes
        )


    def get_defaults() -> dict[str, TypeInfo]:
        if (function := find_function(frame, code)):
            return {
                param_name: get_type(param.default)
                for param_name, param in inspect.signature(function).parameters.items()
                if param.default != inspect._empty
            }

        return {}


    args = inspect.getargvalues(frame)

    def get_self_type() -> tuple[TypeInfo|None, TypeInfo|None, FunctionType|FunctionDescriptor|None]:
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

    # TODO self_type, like overrides, could just be saved in record_function,
    # and computed only when first recording a function.
    self_type, self_replacement, overrides = get_self_type()
    function_data = find_function(frame, code)
    obs.record_function(code, args, get_defaults, overrides, function_data)

    arg_values = (
        *(get_type(args.locals[arg_name]) for arg_name in args.args),
        *(
            (TypeInfo.from_set({
                get_type(val) for val in args.locals[args.varargs]
            }),)
            if args.varargs else ()
        ),
        *(
            (TypeInfo.from_set({
                get_type(val) for val in args.locals[args.keywords].values()
            }),)
            if args.keywords else ()
        )
    )

    obs.record_start(
        code,
        FrameId(id(frame)),
        arg_values,
        self_type,
        self_replacement,
        function_data
    )


instrumentation_functions_code = {
    enter_handler.__code__,
    call_handler.__code__,
    return_handler.__code__,
    yield_handler.__code__,
    send_handler.__code__
}

def in_instrumentation_code() -> bool:
    for frame in sys._current_frames().values():
        # We stop walking the stack after a given number of frames to
        # limit overhead. The instrumentation code should be fairly
        # shallow, so this heuristic should have no impact on accuracy
        # while improving performance.
        f: FrameType|None = frame
        countdown = 10
        while f and countdown > 0:
            # using torch dynamo, f_code can apparently be a dict...
            if isinstance(f.f_code, CodeType) and f.f_code in instrumentation_functions_code:
                # In instrumentation code
                return True
                break
            f = f.f_back
            countdown -= 1
    return False


instrumentation_overhead = 0.0
sample_count_instrumentation = 0.0
sample_count_total = 0.0

def restart_sampling() -> None:
    """
    Measures the instrumentation overhead, restarting event delivery
    if it lies below the target overhead.
    """
    # Walk the stack to see if righttyper instrumentation is running (and thus instrumentation).
    # We use this information to estimate instrumentation overhead, and put off restarting
    # instrumentation until overhead drops below the target threshold.
    global sample_count_instrumentation, sample_count_total
    global instrumentation_overhead
    sample_count_total += 1.0
    if in_instrumentation_code():
        sample_count_instrumentation += 1.0
    instrumentation_overhead = (
        sample_count_instrumentation / sample_count_total
    )
    if instrumentation_overhead <= options.target_overhead / 100.0:
        # Instrumentation overhead is low enough: restart instrumentation.
        sys.monitoring.restart_events()


def execute_script_or_module(
    script: str,
    is_module: bool,
    args: list[str],
) -> None:
    try:
        sys.argv = [script, *args]
        if is_module:
            with loader.ImportManager(replace_dict=options.replace_dict):
                runpy.run_module(
                    script,
                    run_name="__main__",
                    alter_sys=True,
                )
        else:
            with loader.ImportManager(replace_dict=options.replace_dict):
                runpy.run_path(script, run_name="__main__")

    except SystemExit as e:
        if e.code not in (None, 0):
            raise


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


def post_process() -> None:
    sig_changes = process_all_files()

    with open(f"{TOOL_NAME}.out", "w+") as f:
        output_signatures(sig_changes, f)


def process_file_wrapper(args) -> SignatureChanges:
    return process_file(*args)


def process_all_files() -> list[SignatureChanges]:
    fnames = set(
        t.func_id.file_name
        for t in obs.functions_visited.values()
        if not skip_this_file(
            t.func_id.file_name,
            options.script_dir,
            options.include_all,
            options.include_files_pattern
        )
    )

    if len(fnames) == 0:
        return []

    type_annotations = obs.collect_annotations()
    module_names = [*sys.modules.keys(), get_main_module_fqn()]

    args_gen = (
        (
            fname,
            options.output_files,
            options.generate_stubs,
            type_annotations,
            options.overwrite,
            module_names,
            options.ignore_annotations,
            options.only_update_annotations,
            options.inline_generics
        )
        for fname in fnames
    )

    if options.use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(process_file_wrapper, args_gen)
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
        task1 = progress.add_task(description="", total=len(fnames))

        for result in results:
            sig_changes.append(result)
            progress.update(task1, advance=1)
            progress.refresh()

    return sig_changes


FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(
    filename="righttyper.log",
    level=logging.INFO,
    format=FORMAT,
)
logger = logging.getLogger("righttyper")


class CheckModule(click.ParamType):
    name = "module"

    def convert(self, value: str, param: Any, ctx: Any) -> str:
        # Check if it's a valid file path
        if importlib.util.find_spec(value):
            return value

        self.fail(
            f"{value} isn't a valid module",
            param,
            ctx,
        )
        return ""


@click.command(
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
    type=CheckModule(),
)
@click.option(
    "--all-files",
    is_flag=True,
    help="Process any files encountered, including in libraries (except for those specified in --include-files)",
)
@click.option(
    "--include-files",
    type=str,
    help="Include only files matching the given pattern.",
)
@click.option(
    "--include-functions",
    multiple=True,
    help="Only annotate functions matching the given pattern.",
)
@click.option(
    "--infer-shapes",
    is_flag=True,
    default=False,
    show_default=True,
    help="Produce tensor shape annotations (compatible with jaxtyping).",
)
@click.option(
    "--srcdir",
    type=click.Path(exists=True, file_okay=False),
    default=os.getcwd(),
    help="Use this directory as the base for imports.",
)
@click.option(
    "--overwrite/--no-overwrite",
    help="Overwrite files with type information.",
    default=False,
    show_default=True,
)
@click.option(
    "--output-files/--no-output-files",
    help="Output annotated files (possibly overwriting, if specified).",
    default=False,
    show_default=True,
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
    "--verbose",
    is_flag=True,
    help="Print diagnostic information.",
)
@click.option(
    "--generate-stubs",
    is_flag=True,
    help="Generate stub files (.pyi).",
    default=False,
)
@click.version_option(
    version=importlib.metadata.version(TOOL_NAME),
    prog_name=TOOL_NAME,
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
    hidden=True,
    help="Whether to use multiprocessing.",
)
@click.option(
    "--sampling/--no-sampling",
    default=options.sampling,
    help=f"Whether to sample calls and types or to use every one seen.",
    show_default=True,
)
@click.option(
    "--type-coverage",
    nargs=2,
    type=(
        click.Choice(["by-directory", "by-file", "summary"]),
        click.Path(exists=True, file_okay=True),
    ),
    help="Rather than run a script or module, report a choice of 'by-directory', 'by-file' or 'summary' type annotation coverage for the given path.",
)
@click.option(
    "--signal-wakeup/--thread-wakeup",
    default=not platform.system() == "Windows",
    hidden=True,
    help="Whether to use signal-based wakeups or thead-based wakeups."
)
@click.option(
    "--replace-dict/--no-replace-dict",
    is_flag=True,
    help="Whether to replace 'dict' to enable efficient, statistically correct samples."
)
@click.option(
    "--container-sample-limit",
    type=int,
    default=options.container_sample_limit,
    help="Number of container elements to sample.",
)
@click.option(
    "--python-version",
    type=click.Choice(["3.9", "3.10", "3.11", "3.12", "3.13"]),
    default="3.12",
    help="Python version for which to emit annotations.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main(
    script: str,
    module: str,
    args: list[str],
    all_files: bool,
    include_files: str,
    include_functions: tuple[str, ...],
    verbose: bool,
    overwrite: bool,
    output_files: bool,
    ignore_annotations: bool,
    only_update_annotations: bool,
    generate_stubs: bool,
    infer_shapes: bool,
    srcdir: str,
    target_overhead: float,
    use_multiprocessing: bool,
    sampling: bool,
    type_coverage: tuple[str, str],
    signal_wakeup: bool,
    replace_dict: bool,
    container_sample_limit: int,
    python_version: str|tuple[int, ...]
) -> None:

    if type_coverage:
        from . import annotation_coverage as cov
        cov_type, path = type_coverage

        cache = cov.analyze_all_directories(path)

        if cov_type == "by-directory":
            cov.print_directory_summary(cache)
        elif cov_type == "by-file":
            cov.print_file_summary(cache)
        else:
            cov.print_annotation_summary()

        return

    if module:
        args = [*((script,) if script else ()), *args]  # script, if any, is really the 1st module arg
        script = module
    elif script:
        if not os.path.isfile(script):
            raise click.UsageError(f"\"{script}\" is not a file.")
    else:
        raise click.UsageError("Either -m/--module must be provided, or a script be passed.")

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

    python_version = tuple(int(n) for n in python_version.split('.'))

    debug_print_set_level(verbose)
    options.script_dir = os.path.dirname(os.path.realpath(script))
    options.include_files_pattern = include_files
    options.include_all = all_files
    options.include_functions_pattern = include_functions
    options.target_overhead = target_overhead
    options.infer_shapes = infer_shapes
    options.ignore_annotations = ignore_annotations
    options.only_update_annotations = only_update_annotations
    options.overwrite = overwrite
    options.output_files = output_files
    options.generate_stubs = generate_stubs
    options.srcdir = srcdir
    options.use_multiprocessing = use_multiprocessing
    options.sampling = sampling
    options.replace_dict = replace_dict
    options.container_sample_limit = container_sample_limit
    options.use_typing_union = python_version < (3, 10)
    options.use_typing_self = python_version >= (3, 11)
    options.use_typing_never = python_version >= (3, 11)
    options.inline_generics = python_version >= (3, 12)

    alarm_cls = SignalAlarm if signal_wakeup else ThreadAlarm
    alarm = alarm_cls(restart_sampling, 0.01)

    try:
        setup_tool_id()
        register_monitoring_callbacks(
            enter_handler,
            return_handler,
            yield_handler,
            call_handler,
        )
        sys.monitoring.restart_events()
        alarm.start()
        execute_script_or_module(script, bool(module), args)
    finally:
        reset_monitoring()
        alarm.stop()
        post_process()
