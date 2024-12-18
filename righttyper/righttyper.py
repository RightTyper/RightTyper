import concurrent.futures
import functools
import importlib.metadata
import importlib.util
import inspect
import itertools
import logging
import os
import runpy
import signal
import sys
import collections.abc as abc
from collections import defaultdict
from dataclasses import dataclass, field
from types import CodeType, FrameType, FunctionType
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
    get_full_type,
    should_skip_function,
    update_argtypes,
)
from righttyper.righttyper_tool import (
    register_monitoring_callbacks,
    reset_monitoring,
    setup_timer,
    setup_tool_id,
)
from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    Filename,
    FuncInfo,
    FuncAnnotation,
    FunctionName,
    Typename,
    TypeInfo,
    TypeInfoSet,
)
from righttyper.righttyper_utils import (
    TOOL_ID,
    TOOL_NAME,
    debug_print,
    debug_print_set_level,
    skip_this_file,
    union_typeset_str,
    get_main_module_fqn
)

@dataclass
class Options:
    script_dir: str = ""
    include_files_regex: str = ""
    include_all: bool = False
    target_overhead: float = 5.0
    infer_shapes: bool = False
    ignore_annotations: bool = False
    overwrite: bool = False
    output_files: bool = False
    generate_stubs: bool = False
    srcdir: str = ""
    use_multiprocessing: bool = True
    sampling: bool = True

options = Options()


instrumentation_overhead = 0.0
alpha = 0.9
sample_count_instrumentation = 0.0
sample_count_total = 0.0

logger = logging.getLogger("righttyper")


@dataclass
class Observations:
    # All visited functions (file name and function name)
    visited_funcs: set[FuncInfo] = field(default_factory=set)

    # For each visited function, all the info about its arguments
    visited_funcs_arguments: dict[FuncInfo, list[ArgInfo]] = field(default_factory=lambda: defaultdict(list))

    # For each visited function, the values it returned
    visited_funcs_retval: dict[FuncInfo, TypeInfoSet] = field(default_factory=lambda: defaultdict(TypeInfoSet))

    # For each visited function, the values it yielded
    visited_funcs_yieldval: dict[FuncInfo, TypeInfoSet] = field(default_factory=lambda: defaultdict(TypeInfoSet))

    namespace: dict[str, Any] = field(default_factory=dict)

    def _transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this class."""

        def transform_set(s: TypeInfoSet) -> None:
            """Applies the transformer to a set, only replacing items where necessary."""
            for t in list(s):
                tprime = tr.visit(t)
                if t is not tprime:
                    s.remove(t)
                    s.add(tprime)

        for args in self.visited_funcs_arguments.values():
            for i in range(len(args)):
                transform_set(args[i].type_set)

        for ts in itertools.chain(
            self.visited_funcs_yieldval.values(),
            self.visited_funcs_retval.values()
        ):
            transform_set(ts)


    def return_type(self: Self, f: FuncInfo) -> Typename:
        """Returns the return type for a given function."""

        if f in self.visited_funcs_yieldval:
            is_async = False
            y = union_typeset_str(self.visited_funcs_yieldval[f])
            if y == "builtins.async_generator_wrapped_value":
                is_async = True
                y = Typename("typing.Any") # how to unwrap the value without waiting on it?

            r = union_typeset_str(self.visited_funcs_retval[f])

            if is_async:
                # FIXME capture send type and switch to AsyncGenerator if any sent
                return Typename(f"typing.AsyncIterator[{y}]")

            if r == "None":
                # Note that we are unable to differentiate between an implicit "None"
                # return and an explicit "return None".
                return Typename(f"typing.Iterator[{y}]")

            return Typename(f"typing.Generator[{y}, typing.Any, {r}]")

        if f in self.visited_funcs_retval:
            return union_typeset_str(self.visited_funcs_retval[f])

        return Typename("None")


    def collect_annotations(self: Self) -> dict[FuncInfo, FuncAnnotation]:
        """Collects function type annotations from the observed types."""

        class T(TypeInfo.Transformer):
            """Updates Callable type declarations based on observations."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.func and not node.args:
                    if node.func in self.visited_funcs:
                        return TypeInfo('typing', 'Callable', args = (
                                "[" + ", ".join(
                                    union_typeset_str(arg.type_set)
                                    for arg in self.visited_funcs_arguments[node.func][int(node.is_bound):]
                                ) + "]",
                                self.return_type(node.func)
                            )
                        )

                return super().visit(node)

        self._transform_types(T())

        type_annotations: dict[FuncInfo, FuncAnnotation] = {}
        for t in self.visited_funcs:
            args = self.visited_funcs_arguments[t]

            type_annotations[t] = FuncAnnotation(
                [
                    (
                        arginfo.arg_name,
                        union_typeset_str(arginfo.type_set)
                    )
                    for arginfo in args
                ],
                self.return_type(t)
            )

        return type_annotations


    def update_visited_funcs_arguments(
        self: Self,
        t: FuncInfo,
        argtypes: list[ArgInfo]
    ) -> None:
        if t in self.visited_funcs_arguments:
            for i, arginfo in enumerate(argtypes):
                if i < len(self.visited_funcs_arguments[t]):
                    self.visited_funcs_arguments[t][i].type_set.update(arginfo.type_set)
                    # reset_sampling_interval() if all new
        else:
            self.visited_funcs_arguments[t] = argtypes


obs = Observations()


def enter_function(code: CodeType, offset: int) -> Any:
    """
    Process the function entry point, perform monitoring related operations,
    and manage the profiling of function execution.
    """
    if should_skip_function(
        code,
        options.script_dir,
        options.include_all,
        options.include_files_regex,
    ):
        return sys.monitoring.DISABLE

    t = FuncInfo(
        Filename(code.co_filename),
        FunctionName(code.co_qualname),
    )
    obs.visited_funcs.add(t)

    frame = inspect.currentframe()
    if frame and frame.f_back:
        # NOTE: this backtracking logic is brittle and must be
        # adjusted if the call chain changes length.
        frame = frame.f_back
        assert code == frame.f_code

        if function := next(find_functions(frame, code), None):
            defaults = {
                param_name: [param.default]
                for param_name, param in inspect.signature(function).parameters.items()
                if param.default != inspect._empty
            }
        else:
            defaults = {}

        process_function_arguments(t, inspect.getargvalues(frame), defaults)
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
            options.include_files_regex,
        ):
            sys.monitoring.set_local_events(
                TOOL_ID,
                callable.__code__,
                sys.monitoring.events.PY_START
                | sys.monitoring.events.PY_RETURN
                | sys.monitoring.events.PY_YIELD,
            )

    return sys.monitoring.DISABLE


def yield_function(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
) -> object:
    # We do the same thing for yields and exits.
    return exit_function_worker(
        code,
        instruction_offset,
        return_value,
        sys.monitoring.events.PY_YIELD,
    )


def exit_function(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
) -> object:
    return exit_function_worker(
        code,
        instruction_offset,
        return_value,
        sys.monitoring.events.PY_RETURN,
    )


def exit_function_worker(
    code: CodeType,
    instruction_offset: int,
    return_value: Any,
    event_type: int,
) -> object:
    """
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
        options.include_files_regex,
    ):
        return sys.monitoring.DISABLE

    t = FuncInfo(
        Filename(code.co_filename),
        FunctionName(code.co_qualname),
    )

    debug_print(f"exit processing, retval was {obs.visited_funcs_retval[t]=}")

    typeinfo = get_full_type(return_value, use_jaxtyping=options.infer_shapes)

    if event_type == sys.monitoring.events.PY_YIELD:
        obs.visited_funcs_yieldval[t].add(typeinfo)
    else:
        obs.visited_funcs_retval[t].add(typeinfo)

    return sys.monitoring.DISABLE if options.sampling else None


def process_function_arguments(
    t: FuncInfo,
    args: inspect.ArgInfo,
    defaults: dict[str, Any]
) -> None:
    if args.varargs:
        args.args.append(args.varargs)
    if args.keywords:
        args.args.append(args.keywords)

    argtypes: list[ArgInfo] = []
    for arg in args.args:
        if arg:
            update_argtypes(
                argtypes,
                (t, ArgumentName(arg)),
                [args.locals[arg], *defaults.get(arg, [])],
                arg,
                is_vararg = (arg == args.varargs),
                is_kwarg = (arg == args.keywords),
                use_jaxtyping = options.infer_shapes
            )

    debug_print(f"processing {t=} {argtypes=}")
    obs.update_visited_funcs_arguments(t, argtypes)


def find_functions(
    caller_frame: FrameType,
    code: CodeType
) -> abc.Iterator[abc.Callable]:
    """
    Attempts to map back from a code object to the functions that use it.
    """

    visited_wrapped = set()
    visited_classes = set()
    
    def check_function(name: str, obj: abc.Callable) -> abc.Iterator[abc.Callable]:
        while hasattr(obj, "__wrapped__"):
            if obj in visited_wrapped:
                break
            visited_wrapped.add(obj)
            obj = obj.__wrapped__
        if hasattr(obj, "__code__") and obj.__code__ is code:
            yield obj

    def find_in_class(class_obj: object) -> abc.Iterator[abc.Callable]:
        if class_obj in visited_classes:
            return
        visited_classes.add(class_obj)
        for name, obj in class_obj.__dict__.items():
            if inspect.isfunction(obj):
                yield from check_function(name, obj)
            elif inspect.isclass(obj):
                yield from find_in_class(obj)

    dicts: abc.Iterable[tuple[str, Any]] = caller_frame.f_globals.items()
    if caller_frame.f_back:
        dicts = itertools.chain(caller_frame.f_back.f_locals.items(), dicts)

    for name, obj in dicts:
        if inspect.isfunction(obj):
            yield from check_function(name, obj)
        elif inspect.isclass(obj):
            yield from find_in_class(obj)


def in_instrumentation_code(frame: FrameType) -> bool:
    # We stop walking the stack after a given number of frames to
    # limit overhead. The instrumentation code should be fairly
    # shallow, so this heuristic should have no impact on accuracy
    # while improving performance.
    f: FrameType|None = frame
    countdown = 10
    while f and countdown > 0:
        if f.f_code in instrumentation_functions_code:
            # In instrumentation code
            return True
            break
        f = f.f_back
        countdown -= 1
    return False


def restart_sampling(_signum: int, frame: FrameType|None) -> None:
    """
    This function handles the task of clearing the seen functions.
    Called when a timer signal is received.

    Args:
        _signum: The signal number
        _frame: The current stack frame
    """
    # Walk the stack to see if righttyper instrumentation is running (and thus instrumentation).
    # We use this information to estimate instrumentation overhead, and put off restarting
    # instrumentation until overhead drops below the target threshold.
    global sample_count_instrumentation, sample_count_total
    global instrumentation_overhead
    sample_count_total += 1.0
    assert frame is not None
    if in_instrumentation_code(frame):
        sample_count_instrumentation += 1.0
    instrumentation_overhead = (
        sample_count_instrumentation / sample_count_total
    )
    if instrumentation_overhead <= options.target_overhead / 100.0:
        # Instrumentation overhead remains low enough; restart instrumentation.
        # Restart the system monitoring events
        sys.monitoring.restart_events()
    else:
        pass
    # Set a timer for the next round of sampling.
    signal.setitimer(
        signal.ITIMER_REAL,
        0.01,
    )


instrumentation_functions_code = set(
    [
        enter_function.__code__,
        call_handler.__code__,
        exit_function_worker.__code__,
        restart_sampling.__code__,
    ]
)


def execute_script_or_module(
    script: str,
    module: bool,
    args: list[str],
) -> None:
    obs.namespace = {}
    try:
        sys.argv = [script, *args]
        if module:
            obs.namespace = runpy.run_module(
                script,
                run_name="__main__",
                alter_sys=True,
            )
        else:
            obs.namespace = runpy.run_path(script, run_name="__main__")

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
        t.file_name
        for t in obs.visited_funcs
        if not skip_this_file(
            t.file_name,
            options.script_dir,
            options.include_all,
            options.include_files_regex,
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
    help="Include only files matching the given regex pattern.",
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
@click.option(
    "--type-coverage-by-directory",
    type=click.Path(exists=True, file_okay=True),
    help="Report per-directory type annotation coverage for a single file or all Python files in a directory and its children.",
)
@click.option(
    "--type-coverage-by-file",
    type=click.Path(exists=True, file_okay=True),
    help="Report per-file type annotation coverage for a single file or all Python files in a directory and its children.",
)
@click.option(
    "--type-coverage-summary",
    type=click.Path(exists=True, file_okay=True),
    help="Report uncovered and partially covered files and functions when performing type annotation coverage analysis.",
)  # Note: should only be available if coverage-by-directory or coverage-by-file are specified
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
    hidden=True,
    help="Whether to sample calls and types or to use every one seen.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main(
    script: str,
    module: str,
    args: list[str],
    all_files: bool,
    include_files: str,
    type_coverage_by_directory: str,
    type_coverage_by_file: str,
    type_coverage_summary: str,
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
) -> None:

    if module:
        args = [*((script,) if script else ()), *args]  # script, if any, is really the 1st module arg
        script = module
    elif script:
        if not os.path.isfile(script):
            raise click.UsageError(f"\"{script}\" is not a file.")
    else:
        raise click.UsageError(f"Either -m/--module must be provided, or a script be passed.")

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

    if (
        type_coverage_by_directory or type_coverage_by_file
    ) and type_coverage_summary:
        raise click.UsageError(
            'The "--type-coverage-summary" option can only be specified when "--type-coverage-by-directory" or "--type-coverage-by-file" are NOT specified.'
        )

    from . import annotation_coverage

    if type_coverage_summary:
        directory_summary = annotation_coverage.analyze_all_directories(
            type_coverage_summary
        )
        annotation_coverage.print_annotation_summary()
        return

    if type_coverage_by_directory:
        directory_summary = annotation_coverage.analyze_all_directories(
            type_coverage_by_directory
        )
        annotation_coverage.print_directory_summary(directory_summary)
        return

    if type_coverage_by_file:
        file_summary = annotation_coverage.analyze_all_directories(
            type_coverage_by_file
        )
        annotation_coverage.print_file_summary(file_summary)
        return

    debug_print_set_level(verbose)
    options.script_dir = os.path.dirname(os.path.realpath(script))
    options.include_files_regex = include_files
    options.include_all = all_files
    options.target_overhead = target_overhead
    options.infer_shapes = infer_shapes
    options.ignore_annotations = ignore_annotations
    options.overwrite = overwrite
    options.output_files = output_files
    options.generate_stubs = generate_stubs
    options.srcdir = srcdir
    options.use_multiprocessing = use_multiprocessing
    options.sampling = sampling 

    setup_tool_id()
    register_monitoring_callbacks(
        enter_function,
        call_handler,
        exit_function,
        yield_function,
    )
    sys.monitoring.restart_events()
    setup_timer(restart_sampling)
    # replace_dicts.replace_dicts()
    execute_script_or_module(script, bool(module), args)
    reset_monitoring()
    post_process()
