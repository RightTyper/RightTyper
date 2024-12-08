import functools
import importlib.metadata
import importlib.util
import inspect
import itertools
import logging
import multiprocessing
import multiprocessing.connection
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
    get_type_hints,
    Self
)

import click

# Disabled for now
# from righttyper import replace_dicts
from righttyper.righttyper_process import (
    process_file,
)
from righttyper.righttyper_runtime import (
    format_annotation,
    format_function_definition,
    get_adjusted_full_type,
    get_class_type_from_stack,
    should_skip_function,
    update_argtypes,
    get_main_module_fqn
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
    TypenameSet,
)
from righttyper.righttyper_utils import (  # get_sampling_interval,; update_sampling_interval,; reset_sampling_interval,
    TOOL_ID,
    TOOL_NAME,
    debug_print,
    debug_print_set_level,
    make_type_signature,
    skip_this_file,
    unannotated,
    union_typeset_str,
)

# Below is to mollify mypy.
try:
    import sys.monitoring  # pyright: ignore
except Exception:
    pass


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

    # The existing spec for each function (that is, function prototype)
    existing_spec: dict[FuncInfo, str] = field(default_factory=dict)

    # For each visited function, all the info about its arguments
    visited_funcs_arguments: dict[FuncInfo, list[ArgInfo]] = field(default_factory=lambda: defaultdict(list))

    # For each visited function, the values it returned
    visited_funcs_retval: dict[FuncInfo, TypenameSet] = field(default_factory=lambda: defaultdict(TypenameSet))

    # For each visited function, the values it yielded
    visited_funcs_yieldval: dict[FuncInfo, TypenameSet] = field(default_factory=lambda: defaultdict(TypenameSet))

    # For each function and argument, what type the argument is (e.g.,
    # kwarg, vararg)
    arg_types: dict[
        tuple[FuncInfo, ArgumentName],
        ArgumentType,
    ] = field(default_factory=dict)

    # Existing annotations (variable to type annotations, optionally
    # including 'return')
    existing_annotations: dict[FuncInfo, dict[ArgumentName, str]] = field(default_factory=lambda: defaultdict(dict))

    namespace: dict[str, Any] = field(default_factory=dict)


    def collect_annotations(self: Self) -> dict[FuncInfo, FuncAnnotation]:
        """Collects function type annotations from the observed types."""

        type_annotations: dict[FuncInfo, FuncAnnotation] = {}
        for t in self.visited_funcs:
            args = self.visited_funcs_arguments[t]
            arg_annotations = [
                (
                    ArgumentName(arginfo.arg_name),
                    union_typeset_str(
                        arginfo.type_name_set,
                        self.namespace,
                    ),
                )
                for arginfo in args
            ]

            if t in self.visited_funcs_yieldval:
                is_async = False
                y = union_typeset_str(self.visited_funcs_yieldval[t])
                if y == "builtins.async_generator_wrapped_value":
                    is_async = True
                    y = Typename("typing.Any") # how to unwrap the value without waiting on it?

                r = union_typeset_str(self.visited_funcs_retval.get(t, TypenameSet([Typename("None")])))

                if is_async:
                    # FIXME capture send type and switch to AsyncGenerator if any sent
                    retval = f"typing.AsyncIterator[{y}]"
                elif r == "None":
                    # Note that we are unable to differentiate between an implicit "None"
                    # return and an explicit "return None".
                    retval = f"typing.Iterator[{y}]"
                else:
                    retval = f"typing.Generator[{y}, typing.Any, {r}]"

            elif t in self.visited_funcs_retval:
                retval = union_typeset_str(
                    self.visited_funcs_retval[t],
                    self.namespace,
                )
            else:
                retval = "None"
            type_annotations[t] = FuncAnnotation(
                arg_annotations,
                Typename(retval),
            )
            # print(f"{type_annotations[t]} {t}")
        return type_annotations

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
    if frame and frame.f_back and frame.f_back.f_back:
        process_function_arguments(frame, t)

    return sys.monitoring.DISABLE


def call_handler(
    code: CodeType,
    instruction_offset: int,
    callable: object,
    arg0: object,
) -> Any:
    # If we are calling a function, activate its start, return, and yield handlers.
    if isinstance(callable, FunctionType):
        if should_skip_function(
            code,
            options.script_dir,
            options.include_all,
            options.include_files_regex,
        ):
            return sys.monitoring.DISABLE
        try:
            sys.monitoring.set_local_events(
                TOOL_ID,
                callable.__code__,
                sys.monitoring.events.PY_START
                | sys.monitoring.events.PY_RETURN
                | sys.monitoring.events.PY_YIELD,
            )
        except AttributeError:
            pass
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
    func_name = code.co_qualname
    filename = code.co_filename
    if should_skip_function(
        code,
        options.script_dir,
        options.include_all,
        options.include_files_regex,
    ):
        return sys.monitoring.DISABLE

    t = FuncInfo(
        Filename(filename),
        FunctionName(func_name),
    )

    debug_print(f"exit processing, retval was {obs.visited_funcs_retval[t]=}")

    class_type = get_class_type_from_stack()
    typename = Typename(
        get_adjusted_full_type(return_value, class_type, use_jaxtyping=options.infer_shapes)
    )

    if event_type == sys.monitoring.events.PY_YIELD:
        obs.visited_funcs_yieldval[t].update([typename])
    else:
        obs.visited_funcs_retval[t].update([typename])

    return sys.monitoring.DISABLE


def process_function_arguments(
    frame: Any,
    t: FuncInfo,
) -> None:
    # NOTE: this backtracking logic is brittle and must be
    # adjusted if the call chain changes length.
    caller_frame = frame.f_back #.f_back
    code = caller_frame.f_code
    class_type = get_class_type_from_stack()
    arg_names, vararg, kwarg, the_values = inspect.getargvalues(caller_frame)
    if vararg:
        arg_names.append(vararg)
    if kwarg:
        arg_names.append(kwarg)

    type_hints = get_function_type_hints(caller_frame, code)

    update_function_annotations(
        t,
        caller_frame,
        arg_names,
        type_hints,
    )

    # also "observe" any default values
    try:
        _, function = next(find_functions(caller_frame, code))
        defaults = {
            param_name: [param.default]
            for param_name, param in inspect.signature(function).parameters.items()
            if param.default != inspect._empty
        }
    except StopIteration:
        defaults = {}

    argtypes: list[ArgInfo] = []
    for arg in arg_names:
        if arg:
            index = (
                FuncInfo(
                    Filename(caller_frame.f_code.co_filename),
                    FunctionName(code.co_qualname),
                ),
                ArgumentName(arg),
            )
            update_argtypes(
                argtypes,
                obs.arg_types,
                index,
                [the_values[arg], *defaults.get(arg, [])],
                class_type,
                arg,
                is_vararg = (arg == vararg),
                is_kwarg = (arg == kwarg),
                use_jaxtyping = options.infer_shapes
            )

    debug_print(f"processing {t=} {argtypes=}")
    update_visited_funcs_arguments(t, argtypes)


def find_functions(
    caller_frame: FrameType,
    code: CodeType
) -> abc.Iterator[tuple[str, abc.Callable]]:
    """
    Attempts to map back from a code object to the functions that use it.
    """

    visited_wrapped = set()
    visited_classes = set()
    
    def check_function(name: str, obj: abc.Callable) -> abc.Iterator[tuple[str, abc.Callable]]:
        while hasattr(obj, "__wrapped__"):
            if obj in visited_wrapped:
                break
            visited_wrapped.add(obj)
            obj = obj.__wrapped__
        if hasattr(obj, "__code__") and obj.__code__ is code:
            yield (name, obj)

    def find_in_class(class_obj: object) -> abc.Iterator[tuple[str, abc.Callable]]:
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


@functools.cache
def get_function_type_hints(
    caller_frame: Any,
    code: CodeType,
) -> dict[str, str]:
    for (
        name,
        obj,
    ) in find_functions(caller_frame, code):
        try:
            return get_type_hints(obj)
        except Exception:
            return {}
    return {}


def update_function_annotations(
    t: FuncInfo,
    caller_frame: Any,
    args: list[str],
    type_hints: dict[str, str],
) -> None:

    for (
        name,
        obj,
    ) in find_functions(caller_frame, caller_frame.f_code):
        # TODO stop at the first?
        obs.existing_spec[t] = format_function_definition(
            name,
            args,
            (type_hints if not options.ignore_annotations else {}),
        )
        obs.existing_annotations[t] = {
            ArgumentName(name): format_annotation(type_hints[name])
            for name in type_hints
        }


def update_visited_funcs_arguments(
    t: FuncInfo, argtypes: list[ArgInfo]
) -> None:
    if t in obs.visited_funcs_arguments:
        for i, arginfo in enumerate(argtypes):
            if i < len(obs.visited_funcs_arguments[t]):
                obs.visited_funcs_arguments[t][i].type_name_set.update(arginfo.type_name_set)
                # reset_sampling_interval() if all new
    else:
        obs.visited_funcs_arguments[t] = argtypes


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


def output_type_signatures(
    file: TextIO = sys.stdout,
    namespace: dict[str, Any] = globals(),
) -> None:
    # Print all type signatures
    fname_printed: dict[Filename, bool] = defaultdict(bool)
    visited_funcs_by_fname = sorted(
        obs.visited_funcs, key=lambda a: a.file_name + ":" + a.func_name
    )
    for t in visited_funcs_by_fname:
        if skip_this_file(
            t.file_name,
            options.script_dir,
            options.include_all,
            options.include_files_regex,
        ):
            continue
        try:
            s = make_type_signature(
                file_name=t.file_name,
                func_name=t.func_name,
                args=obs.visited_funcs_arguments[t],
                retval=obs.visited_funcs_retval[t],
                namespace=namespace,
                arg_types=obs.arg_types,
                existing_annotations=obs.existing_annotations,
            )
            if t in obs.existing_spec and s == obs.existing_spec[t]:
                continue
            if not fname_printed[t.file_name]:
                print(
                    f"{t.file_name}:\n{'=' * (len(t.file_name) + 1)}\n",
                    file=file,
                )
                fname_printed[t.file_name] = True
            print(f"{s} ...\n", file=file)
            # Print diffs
            if t in obs.existing_spec:
                import difflib

                diffs = difflib.ndiff(
                    (obs.existing_spec[t] + "\n").splitlines(True),
                    (s + "\n").splitlines(True),
                )
                print("".join(diffs), file=file)

        except KeyError:
            # Something weird happened
            logger.exception(f"KeyError: {t=}")


def execute_script_or_module(
    script: str,
    module: bool,
    tool_args: list[str],
    script_args: list[str],
) -> None:
    obs.namespace = {}
    if module:
        sys.argv = [script] + tool_args + script_args
        try:
            obs.namespace = runpy.run_module(
                script,
                run_name="__main__",
                alter_sys=True,
            )
        except SystemExit:
            pass
    else:
        sys.argv = [script] + tool_args + script_args
        obs.namespace = runpy.run_path(script, run_name="__main__")


def post_process() -> None:
    output_type_signatures_to_file(obs.namespace)
    if options.output_files or options.generate_stubs:
        process_all_files()


def output_type_signatures_to_file(namespace: dict[str, Any]) -> None:
    with open(f"{TOOL_NAME}.out", "w+") as f:
        output_type_signatures(f, namespace)


def process_all_files() -> None:
    module_names=[*sys.modules.keys(), get_main_module_fqn()]

    processes: list[multiprocessing.Process] = []
    all_files = list(set(t.file_name for t in obs.visited_funcs))
    prefix = os.path.commonprefix(list(all_files))

    # Collect the file names to process
    fnames_set = set()
    for t in obs.visited_funcs:
        fname = t.file_name
        if fname and should_update_file(t, fname, prefix, obs.namespace):
            assert not skip_this_file(
                fname,
                options.script_dir,
                options.include_all,
                options.include_files_regex,
            )
            fnames_set.add(fname)

    if len(fnames_set) == 0:
        return

    fnames = list(fnames_set)
    ### multiprocessing.set_start_method('fork')
    import rich.progress
    from rich.table import Column

    with rich.progress.Progress(  # rich.progress.TextColumn("[progress.description]{task.description}", table_column=Column(ratio=1)),
        rich.progress.BarColumn(table_column=Column(ratio=1)),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        transient=True,
        expand=True,
        auto_refresh=False,
    ) as progress:
        task1 = progress.add_task(description="", total=len(fnames))
        type_annotations = obs.collect_annotations()
        for fname in fnames:
            args = (
                fname,
                options.output_files,
                options.generate_stubs,
                type_annotations,
                options.overwrite,
                module_names,
                options.ignore_annotations,
                options.srcdir,
            )
            if options.use_multiprocessing:
                process = multiprocessing.Process(
                    target=process_file, args=args
                )
                processes.append(process)
                process.start()
            else:
                process_file(*args)
                progress.update(task1, advance=1)
                progress.refresh()

        if options.use_multiprocessing:
            sentinels = [p.sentinel for p in processes]
            total = len(processes)
            completed = 0
            progress.start()
            while completed < total:
                ready_sentinels = multiprocessing.connection.wait(
                    sentinels
                )  # Wait for any process sentinel to become ready

                for sentinel in ready_sentinels:
                    if isinstance(
                        sentinel, int
                    ):  # should be true for all, mollifying mypy
                        completed += 1
                        progress.update(task1, advance=1)
                        progress.refresh()
                        # Remove the sentinel to avoid double counting
                        sentinels.remove(sentinel)
            # Ensure all processes have finished
            for process in processes:
                process.join()
            progress.update(task1, completed=total)


def should_update_file(
    t: FuncInfo,
    fname: str,
    prefix: str,
    namespace: dict[str, Any],
) -> bool:
    try:
        s = make_type_signature(
            file_name=t.file_name,
            func_name=t.func_name,
            args=obs.visited_funcs_arguments[t],
            retval=obs.visited_funcs_retval[t],
            namespace=namespace,
            arg_types=obs.arg_types,
            existing_annotations=obs.existing_annotations,
        )
    except KeyError:
        return False
    return t not in obs.existing_spec or s != obs.existing_spec[t]


FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(
    filename="righttyper.log",
    level=logging.INFO,
    format=FORMAT,
)
logger = logging.getLogger("righttyper")


class ScriptParamType(click.ParamType):
    def convert(self, value: str, param: Any, ctx: Any) -> str:
        # Check if it's a valid file path
        if os.path.isfile(value):
            return value
        # Check if it's an importable module
        try:
            importlib.import_module(value)
            return value
        except ImportError:
            self.fail(
                f"{value} is neither a file nor a module",
                param,
                ctx,
            )
            return ""


def split_args_at_triple_dash(
    args: list[str],
) -> tuple[list[str], list[str]]:
    tool_args = []
    script_args = []
    triple_dash_found = False
    for arg in args:
        if arg == "---":
            triple_dash_found = True
            continue
        if triple_dash_found:
            script_args.append(arg)
        else:
            tool_args.append(arg)
    return tool_args, script_args


SCRIPT = ScriptParamType()


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument("script", type=SCRIPT, required=False)
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
    "-m",
    "--module",
    is_flag=True,
    help="Run the script as a module.",
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
@click.option(
    "---",
    "triple_dash",
    is_flag=True,
    help="Indicator to separate tool arguments from script or module arguments.",
    hidden=True,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
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
def main(
    script: str,
    all_files: bool,
    include_files: str,
    module: bool,
    triple_dash: bool,
    args: list[str],
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
    use_multiprocessing: bool
) -> None:
    """
    RightTyper efficiently generates types for your function
    arguments and return values.
    """
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

    tool_args, script_args = split_args_at_triple_dash(args)
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
    execute_script_or_module(script, module, tool_args, script_args)
    reset_monitoring()
    post_process()
