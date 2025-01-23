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

import collections.abc as abc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType, FunctionType, GeneratorType, AsyncGeneratorType
from typing import (
    Any,
    TextIO,
    Self,
    Callable,
    Sequence
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
    unwrap,
    get_value_type,
    get_type_name,
    should_skip_function,
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
    FunctionName,
    TypeInfo,
    NoneTypeInfo,
    AnyTypeInfo,
    Sample,
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
    inline_generics: bool = False

options = Options()


logger = logging.getLogger("righttyper")

@dataclass
class Observations:
    # Visited functions' and information about them
    functions_visited: dict[CodeId, FuncInfo] = field(default_factory=dict)

    # Started, but not (yet) completed samples
    pending_samples: dict[tuple[CodeId, FrameId], Sample] = field(default_factory=dict)

    # Completed samples
    samples: dict[CodeId, set[tuple[TypeInfo, ...]]] = field(default_factory=lambda: defaultdict(set))


    def record_function(
        self,
        code: CodeType,
        arg_names: tuple[str, ...],
        get_default_type: Callable[[str], TypeInfo|None]
    ) -> None:
        """Records that a function was visited, along with some details about it."""

        code_id = CodeId(id(code))
        if code_id not in self.functions_visited:
            self.functions_visited[code_id] = FuncInfo(
                FuncId(
                    Filename(code.co_filename),
                    code.co_firstlineno,
                    FunctionName(code.co_qualname),
                ),
                tuple(
                    ArgInfo(ArgumentName(name), get_default_type(name))
                    for name in arg_names
                )
            )


    def record_start(
        self,
        code: CodeType,
        frame_id: FrameId,
        arg_types: tuple[TypeInfo, ...],
        self_type: TypeInfo|None
    ) -> None:
        """Records a function start."""

        # print(f"record_start {code.co_qualname} {arg_types}")
        self.pending_samples[(CodeId(id(code)), frame_id)] = Sample(
            arg_types,
            self_type=self_type,
            is_async=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE)),
            is_generator=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)),
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
            self.samples[code_id].add(sample.process())
            del self.pending_samples[(code_id, frame_id)]
            return True

        return False


    def _transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this class."""

        for sample_set in self.samples.values():
            for s in list(sample_set):
                sprime = tuple(tr.visit(t) for t in s)
                if sprime != s:
                    sample_set.remove(s)
                    sample_set.add(sprime)


    def collect_annotations(self: Self) -> dict[FuncId, FuncAnnotation]:
        """Collects function type annotations from the observed types."""

        # Finish samples for any generators that are still unfinished
        # TODO are there other cases we should handle?
        for (code_id, _), sample in self.pending_samples.items():
            if sample.yields:
                self.samples[code_id].add(sample.process())

        def mk_annotation(code_id: CodeId) -> FuncAnnotation|None:
            func_info = self.functions_visited[code_id]
            samples = self.samples[code_id]

            if (signature := generalize(list(samples))) is None:
                print(f"Error generalizing {func_info.func_id}: inconsistent samples.\n" +
                      f"{[tuple(str(t) for t in s) for s in samples]}")
                return None

            # Annotations are pickled by 'multiprocessing', but many type objects
            # (such as local ones, or from __main__) aren't pickleable.
            class RemoveTypeObjTransformer(TypeInfo.Transformer):
                def visit(vself, node: TypeInfo) -> TypeInfo:
                    if node.type_obj:
                        node = node.replace(type_obj=None)
                    return super().visit(node)

            tr = RemoveTypeObjTransformer()

            return FuncAnnotation(
                args=[
                    (
                        arg.arg_name,
                        tr.visit(
                            merged_types({
                                signature[i],
                                *((arg.default,) if arg.default is not None else ())
                            })
                        )
                    )
                    for i, arg in enumerate(func_info.args)
                ],
                retval=tr.visit(signature[-1])
            )

        class T(TypeInfo.Transformer):
            """Updates Callable type declarations based on observations."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                # if 'args' is there, the function is already annotated
                if node.code_id and (options.ignore_annotations or not node.args) and node.code_id in self.samples:
                    if (ann := mk_annotation(node.code_id)):
                        if node.name == 'Callable':
                            # TODO: fix callable arguments being strings
                            return TypeInfo('typing', 'Callable', args=(
                                f"[{", ".join(map(lambda a: str(a[1]), ann.args[int(node.is_bound):]))}]",
                                ann.retval
                            ))
                        elif node.name in ('Generator', 'AsyncGenerator'):
                            return ann.retval
                        elif node.name == 'Coroutine':
                            return node.replace(args=(NoneTypeInfo, NoneTypeInfo, ann.retval))

                return super().visit(node)

        self._transform_types(T())

        return {
            self.functions_visited[code_id].func_id: annotation
            for code_id in self.samples
            if (annotation := mk_annotation(code_id)) is not None
        }


obs = Observations()


def send_handler(code: CodeType, frame_id: FrameId, arg0: Any) -> None:
    obs.record_send(
        code,
        frame_id, 
        get_value_type(arg0, use_jaxtyping=options.infer_shapes)
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
        function = find_function(frame, code)
        process_function_arguments(code, FrameId(id(frame)), inspect.getargvalues(frame), function)
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
        typeinfo = get_value_type(return_value, use_jaxtyping=options.infer_shapes)

        if event_type == sys.monitoring.events.PY_YIELD:
            found = obs.record_yield(code, FrameId(id(frame)), typeinfo)
        else:
            found = obs.record_return(code, FrameId(id(frame)), typeinfo)

        del frame

    # If the frame wasn't found, keep the event enabled, as this event may be from another
    # invocation whose start we missed.
    return sys.monitoring.DISABLE if (options.sampling and found) else None


def process_function_arguments(
    code: CodeType,
    frame_id: FrameId,
    args: inspect.ArgInfo,
    function: Callable|None
) -> None:

    def get_type(v: Any) -> TypeInfo:
        return get_value_type(v, use_jaxtyping=options.infer_shapes)


    defaults: dict[str, tuple[Any]] = {} if not function else {
        # use tuple to differentiate a None default from no default
        param_name: (param.default,)
        for param_name, param in inspect.signature(function).parameters.items()
        if param.default != inspect._empty
    }


    def get_default_type(name: str) -> TypeInfo|None:
        if (def_value := defaults.get(name)):
            return get_type(*def_value)

        return None

        is_property: bool = (
            (attr := getattr(type(args.locals[args.args[0]]), code.co_name, None)) and
            isinstance(attr, property)
        )

    def get_self_type() -> TypeInfo|None:
        if args.args:
            first_arg = args.locals[args.args[0]]

            # @property?
            if isinstance(getattr(type(first_arg), code.co_name, None), property):
                return get_type(first_arg)

            if function:
                # if type(first_arg) is type, we may have a @classmethod
                first_arg_class = first_arg if type(first_arg) is type else type(first_arg)

                for ancestor in first_arg_class.__mro__:
                    if unwrap(ancestor.__dict__.get(function.__name__, None)) is function:
                        if first_arg is first_arg_class:
                            return get_type_name(first_arg)

                        # normal method
                        return get_type(first_arg)
        return None

    obs.record_function(
        code,
        (
            *(a for a in args.args),
            *((args.varargs,) if args.varargs else ()),
            *((args.keywords,) if args.keywords else ())
        ),
        get_default_type
    )

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
        frame_id,
        arg_values,
        get_self_type()
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
            with loader.ImportManager():
                runpy.run_module(
                    script,
                    run_name="__main__",
                    alter_sys=True,
                )
        else:
            with loader.ImportManager():
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


def process_file_wrapper(args) -> SignatureChanges|BaseException:
    try:
        return process_file(*args)
    except BaseException as e:
        return e


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
            options.inline_generics
        )
        for fname in fnames
    )

    def process_files() -> abc.Iterator[SignatureChanges|BaseException]:
        if options.use_multiprocessing:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                yield from executor.map(process_file_wrapper, args_gen)
        else:
            yield from map(process_file_wrapper, args_gen)

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

        exception = None
        for result in process_files():
            if isinstance(result, BaseException):
                exception = result
            else:
                sig_changes.append(result)

            progress.update(task1, advance=1)
            progress.refresh()

        # complete as much of the work as possible before raising
        if exception:
            raise exception

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
    "--inline-generics",
    is_flag=True,
    help="Declare type variables inline for generics rather than separately."
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
    generate_stubs: bool,
    infer_shapes: bool,
    srcdir: str,
    target_overhead: float,
    use_multiprocessing: bool,
    sampling: bool,
    inline_generics: bool,
    type_coverage: tuple[str, str],
    signal_wakeup: bool
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

    debug_print_set_level(verbose)
    options.script_dir = os.path.dirname(os.path.realpath(script))
    options.include_files_pattern = include_files
    options.include_all = all_files
    options.include_functions_pattern = include_functions
    options.target_overhead = target_overhead
    options.infer_shapes = infer_shapes
    options.ignore_annotations = ignore_annotations
    options.overwrite = overwrite
    options.output_files = output_files
    options.generate_stubs = generate_stubs
    options.srcdir = srcdir
    options.use_multiprocessing = use_multiprocessing
    options.sampling = sampling
    options.inline_generics = inline_generics

    alarm_cls = SignalAlarm if signal_wakeup else ThreadAlarm
    alarm = alarm_cls(restart_sampling, 0.01)
    
    try:
        setup_tool_id()
        register_monitoring_callbacks(
            enter_handler,
            call_handler,
            return_handler,
            yield_handler,
        )
        sys.monitoring.restart_events()
        alarm.start()
        execute_script_or_module(script, bool(module), args)
    finally:
        reset_monitoring()
        alarm.stop()
        post_process()
