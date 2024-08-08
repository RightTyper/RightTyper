import click
import inspect
import logging
import multiprocessing
import os
import random
import runpy
import signal
import sys
import time
import importlib.metadata

from collections import defaultdict
from functools import cache
from types import (
    CodeType,
    FrameType,
    FunctionType,
)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    get_type_hints,
)

from righttyper.get_import_details import (
    get_import_details,
)
from righttyper.righttyper_process import (
    collect_data,
    output_stub_files,
    process_file,
)
from righttyper.righttyper_runtime import (
    format_annotation,
    format_function_definition,
    get_adjusted_full_type,
    get_class_name_from_stack,
    get_class_source_file,
    requires_import,
    should_skip_function,
    update_argtypes,
)
from righttyper.righttyper_shapes import (
    print_annotations,
    update_arg_shapes,
    update_retval_shapes,
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
    ExecInfo,
    Filename,
    FuncInfo,
    FunctionName,
    ImportInfo,
    Typename,
    TypenameFrequency,
    TypenameSet,
)
from righttyper.righttyper_utils import (
    TOOL_ID,
    TOOL_NAME,
    debug_print,
    debug_print_set_level,
    get_sampling_interval,
    make_type_signature,
    reset_sampling_interval,
    skip_this_file,
    unannotated,
    update_sampling_interval,
)

# Below is to mollify mypy.
try:
    import sys.monitoring  # pyright: ignore
except Exception:
    pass

start_execution_time = time.perf_counter_ns()
total_instrumentation_time = 0

# We do this to track instrumentation overhead

import functools
def track_instrumentation_overhead(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global total_instrumentation_time  # Declare total_time as nonlocal to modify it inside the wrapper
        start_time = time.perf_counter_ns()  # Record start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter_ns()  # Record end time
        duration = end_time - start_time  # Calculate duration
        total_instrumentation_time += duration  # Update total_time
        return result

    return wrapper


logger = logging.getLogger("righttyper")

# All visited functions (file name and function name)
visited_funcs: Set[FuncInfo] = set()

# All sampled functions (may be cleared, always a subset of visited_funcs).
sampled_funcs: Set[FuncInfo] = set()

# Any function that yielded at least once (and so will be treated as a
# Generator)
yielded_funcs: Set[FuncInfo] = set()

# The existing spec for each function (that is, function prototype)
existing_spec: Dict[FuncInfo, str] = dict()

# For each visited function, all the info about its arguments
visited_funcs_arguments: Dict[FuncInfo, List[ArgInfo]] = dict()

# For each visited function, all the info about its return value
visited_funcs_retval: Dict[FuncInfo, TypenameSet] = defaultdict(
    lambda: TypenameSet(set())
)

# For each function and argument, what type the argument is (e.g.,
# kwarg, vararg)
arg_types: Dict[
    Tuple[FuncInfo, ArgumentName],
    ArgumentType,
] = dict()

# For each function, the variables (and, potentially, 'return') that
# have no type annotations
not_annotated: Dict[FuncInfo, Set[str]] = defaultdict(set)

# Existing annotations (variable to type annotations, optionally
# including 'return')
existing_annotations: Dict[FuncInfo, Dict[str, str]] = defaultdict(dict)

# Thread-local execution time tracking for functions
exec_info = ExecInfo()

# Track the file names and class names for classes
# that will need import statements
imports: Set[ImportInfo] = set()


namespace: Dict[str, Any] = {}
script_dir = ""
include_files_regex = ""
include_all = False

# @track_instrumentation_overhead
def enter_function(ignore_annotations: bool, code: CodeType) -> Any:
    """
    Process the function entry point, perform monitoring related operations,
    and manage the profiling of function execution.

    Args:
        code : CodeType object

    Returns:
        str: Status of monitoring
    """
    if should_skip_function(
        code,
        script_dir,
        include_all,
        include_files_regex,
    ):
        return sys.monitoring.DISABLE

    t = FuncInfo(
        Filename(code.co_filename),
        FunctionName(code.co_qualname),
    )

    # print(f"calling {t}")
    if t in sampled_funcs:
        elapsed_time = time.perf_counter_ns() - start_execution_time
        exec_fn_fraction = total_instrumentation_time / elapsed_time
        if exec_fn_fraction > 0.05: # FIXME
            # print(f"disabling {t} because sampled and too much overhead {exec_fn_fraction}.")
            return sys.monitoring.DISABLE

    # Record the start time of the function, keyed to the current thread
    exec_info.start_time[t].append(time.perf_counter_ns())
    # print(f"[ entering {t} {code} {native_tid=}")

    sampled_funcs.add(t)
    visited_funcs.add(t)
    
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        process_function_arguments(frame, t, ignore_annotations)

    return # sys.monitoring.DISABLE  # was sys.monitoring.events.PY_RETURN


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
            script_dir,
            include_all,
            include_files_regex,
        ):
            return sys.monitoring.DISABLE
        try:
            sampled_funcs.clear()
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


# @track_instrumentation_overhead
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
    code (CodeType): bytecode of the function.
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
        script_dir,
        include_all,
        include_files_regex,
    ):
        return sys.monitoring.DISABLE

    t = FuncInfo(
        Filename(filename),
        FunctionName(func_name),
    )

    # Note: we use ns to avoid rounding issues
    def disable_probability(overhead: int, duration: float, threshold_fraction: float) -> float:
        return min(1.0, overhead / duration * 1.0 /(threshold_fraction))

    # Update execution time
    current_time = time.perf_counter_ns()
    if t in exec_info.start_time and len(exec_info.start_time[t]) > 0:
        exec_time = current_time - exec_info.start_time[t].pop()
        exec_info.execution_time[t].push(exec_time)

    # Special handling for functions that yielded.
    if t in yielded_funcs:
        # FIXME need to merge generator and return type for higher precision
        return sys.monitoring.DISABLE

    # Check if this is a method. If so, we need to replace anything using the class name with Self.
    class_name = get_class_name_from_stack()

    # Initialize if the function is first visited
    if t not in visited_funcs_retval:
        visited_funcs_retval[t] = TypenameSet(set())
    debug_print(f"exit processing, retval was {visited_funcs_retval[t]=}")

    update_retval_shapes(t, return_value)
    typename = get_adjusted_full_type(return_value, class_name)
    if event_type == sys.monitoring.events.PY_YIELD:
        # Yield: call it a generator
        # FIXME: We should be returning more precise Generators if we discover a return value.
        # See https://docs.python.org/3.10/library/typing.html#typing.Generator
        typename = f"Generator[{typename}, Any, Any]"
        yielded_funcs.add(t)

    # Check if the return value type is already in the set
    found = False
    for typename_frequency in visited_funcs_retval[t]:
        if typename_frequency.typename == typename:
            typename_frequency.counter += 1
            found = True
            break

    # If the return value type is not in the set, add it
    if not found:
        visited_funcs_retval[t].add(TypenameFrequency(Typename(typename), 1))

    debug_print(f"exit processing, retval NOW {visited_funcs_retval[t]=}")

    # FIXME: consider limiting wrt total execution time, not just per-function exec time
    try:
        mean_duration = exec_info.execution_time[t].mean()
        assert isinstance(mean_duration, float)
        # FIXME: cost should be calibrated/computed automatically, and
        # threshold should be cmd-line param
        p_dis = disable_probability(2000, mean_duration, 0.05) 
    except:
        mean_duration = 1_000_000
        p_dis = 0


    return 0 # FIXME DISABLED BELOW FUNCTIONALITY

    # Disable with this probability
    if random.random() <= p_dis:
        print(f"DISABLE {p_dis=} {t=} {mean_duration=}")
        return sys.monitoring.DISABLE
    else:
        # print(f"CONTINUE {p_dis=} {t=} {mean_duration=}")
        return


def process_function_arguments(
    frame: Any,
    t: FuncInfo,
    ignore_annotations: bool,
) -> None:
    caller_frame = frame.f_back.f_back
    code = caller_frame.f_code
    class_name = get_class_name_from_stack()
    args, varargs, varkw, the_values = inspect.getargvalues(
        frame.f_back.f_back
    )
    if varargs:
        args.append(varargs)
    if varkw:
        args.append(varkw)

    type_hints = get_function_type_hints(
        caller_frame, code, ignore_annotations
    )
    update_arg_shapes(t, the_values)
    
    update_function_annotations(
        t,
        caller_frame,
        args,
        type_hints,
        ignore_annotations,
    )

    argtypes: List[ArgInfo] = []
    for arg in args:
        if arg:
            index = (
                FuncInfo(Filename(caller_frame.f_code.co_filename),
                         FunctionName(code.co_qualname)),
                ArgumentName(arg),
            )
            update_argtypes(
                argtypes,
                arg_types,
                index,
                the_values,
                class_name,
                arg,
                varargs,
                varkw,
            )

            if requires_import(the_values[arg]) and arg not in [
                "self",
                "cls",
            ]:
                # print(f"trying to add: {arg=} {the_values[arg]=}")
                try:
                    add_new_import(
                        caller_frame.f_code.co_filename,
                        the_values[arg],
                    )
                except TypeError:
                    # Unexpected type error
                    logger.exception(
                        "process_function_arguments - TypeError:"
                        f" {t=} {arg=} {the_values[arg]=}"
                    )

    debug_print(f"processing {t=} {argtypes=}")
    update_visited_funcs_arguments(t, argtypes)


@cache
def get_function_type_hints(
    caller_frame: Any,
    code: CodeType,
    ignore_annotations: bool,
) -> Dict[str, str]:
    for (
        name,
        obj,
    ) in caller_frame.f_globals.items():
        if hasattr(obj, "__wrapped__"):
            obj = obj.__wrapped__
        if inspect.isfunction(obj) and obj.__code__ is code:
            try:
                return get_type_hints(obj)
            except Exception:
                return {}
    return {}


def update_function_annotations(
    t: FuncInfo,
    caller_frame: Any,
    args: List[str],
    type_hints: Dict[str, str],
    ignore_annotations: bool,
) -> None:
    for (
        name,
        obj,
    ) in caller_frame.f_globals.items():
        if hasattr(obj, "__wrapped__"):
            obj = obj.__wrapped__
        if inspect.isfunction(obj) and obj.__code__ is caller_frame.f_code:
            existing_spec[t] = format_function_definition(
                name,
                args,
                (type_hints if not ignore_annotations else {}),
            )
            not_annotated[t] = unannotated(obj, ignore_annotations)
            existing_annotations[t] = {
                name: format_annotation(type_hints[name])
                for name in type_hints
            }


def add_new_import(filename: str, value: Any) -> None:
    class_src_file = get_class_source_file(value.__class__)
    deets = get_import_details(value)
    new_import = ImportInfo(
        Filename(filename),
        Filename(class_src_file),
        value.__class__.__name__,
        deets,
    )
    imports.add(new_import)


def update_visited_funcs_arguments(
    t: FuncInfo, argtypes: List[ArgInfo]
) -> None:
    if t in visited_funcs_arguments:
        for i, arginfo in enumerate(argtypes):
            if i < len(visited_funcs_arguments[t]):
                update_argument_type(
                    visited_funcs_arguments[t][i],
                    arginfo.type_name_set,
                )
    else:
        visited_funcs_arguments[t] = argtypes


def update_argument_type(
    old_arginfo: ArgInfo,
    full_type_name_set: TypenameSet,
) -> None:
    for full_type_name in full_type_name_set:
        if any(
            full_type_name.typename == old_type_name.typename
            for old_type_name in old_arginfo.type_name_set
        ):
            for old_type_name in old_arginfo.type_name_set:
                if full_type_name.typename == old_type_name.typename:
                    old_type_name.counter += 1
                    break
        else:
            old_arginfo.type_name_set.add(next(iter(full_type_name_set)))
            reset_sampling_interval()


def handle_clear_seen_funcs(_signum: int, _frame: Optional[FrameType]) -> None:
    """
    This function handles the task of clearing the seen functions.
    Called when a timer signal is received.

    Args:
        _signum: The signal number
        _frame: The current stack frame
    """
    # Clear the sampled functions
    sampled_funcs.clear()
    # Set a timer for the next round of sampling.
    update_sampling_interval()
    signal.setitimer(
        signal.ITIMER_REAL,
        get_sampling_interval(),
    )
    # Restart the system monitoring events
    sys.monitoring.restart_events()


def output_type_signatures(
    file: TextIO = sys.stdout,
    namespace: Dict[str, Any] = globals(),
) -> None:
    # Print all type signatures
    fname_printed : Dict[Filename, bool] = defaultdict(bool)
    visited_funcs_by_fname = sorted(visited_funcs, key = lambda a: a.file_name + ":" + a.func_name)
    for t in visited_funcs_by_fname:
        if skip_this_file(
            t.file_name,
            script_dir,
            include_all,
            include_files_regex,
        ):
            continue
        try:
            s = make_type_signature(
                file_name=t.file_name,
                func_name=t.func_name,
                args=visited_funcs_arguments[t],
                retval=visited_funcs_retval[t],
                namespace=namespace,
                not_annotated=not_annotated,
                arg_types=arg_types,
                existing_annotations=existing_annotations,
            )
            if t in existing_spec and s == existing_spec[t]:
                continue
            if not fname_printed[t.file_name]:
                print(f"{t.file_name}:\n{'-' * (len(t.file_name) + 1)}\n", file=file)
                fname_printed[t.file_name] = True
            print(f"{s} ...\n", file=file)
            # Print diffs
            import difflib
            diffs = difflib.ndiff((existing_spec[t] + "\n").splitlines(True),
                                  (s + "\n").splitlines(True))
            print(''.join(diffs), file=file)
        except KeyError:
            # Something weird happened
            logger.exception(f"KeyError: {t=}")


def initialize_globals(
    include_files: str,
    _include_all: bool,
    script: str,
    verbose: bool,
) -> None:
    debug_print_set_level(verbose)
    global include_files_regex, include_all, script_dir
    include_files_regex = include_files
    include_all = _include_all
    script_dir = os.path.dirname(os.path.realpath(script))


def execute_script_or_module(
    script: str,
    module: bool,
    tool_args: List[str],
    script_args: List[str],
) -> None:
    global namespace
    namespace = {}
    if module:
        sys.argv = [script] + tool_args
        try:
            namespace = runpy.run_module(
                script,
                run_name="__main__",
                alter_sys=True,
            )
        except SystemExit:
            pass
    else:
        sys.argv = [script] + script_args
        namespace = runpy.run_path(script, run_name="__main__")


def post_process(
    overwrite: bool = True,
    ignore_annotations: bool = False,
    insert_imports: bool = False,
    generate_stubs: bool = False,
    srcdir: str = "",
) -> None:
    global namespace
    output_type_signatures_to_file(namespace)
    # FIXME output_shapes
    print_annotations()
    if generate_stubs:
        output_stub_files(
            namespace,
            srcdir,
            imports,
            visited_funcs,
            script_dir,
            include_all,
            include_files_regex,
            visited_funcs_arguments,
            visited_funcs_retval,
            not_annotated,
            arg_types,
            existing_annotations,
        )
    process_all_files(
        ignore_annotations,
        overwrite,
        insert_imports,
        srcdir,
    )


def output_type_signatures_to_file(namespace: Dict[str, Any]) -> None:
    with open(f"{TOOL_NAME}.out", "w+") as f:
        output_type_signatures(f, namespace)


def process_all_files(
    ignore_annotations: bool,
    overwrite: bool,
    insert_imports: bool,
    srcdir: str,
) -> None:
    use_multiprocessing = True  # False

    processes: List[multiprocessing.Process] = []
    all_files = list(set(t.file_name for t in visited_funcs))
    prefix = os.path.commonprefix(list(all_files))

    # Collect the file names to process
    fnames_set = set()
    for t in visited_funcs:
        fname = t.file_name
        if fname and should_update_file(t, fname, prefix, namespace):
            assert not skip_this_file(
                fname,
                script_dir,
                include_all,
                include_files_regex,
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
    ) as progress:
        task1 = progress.add_task(description="", total=len(fnames))
        for fname in fnames:
            type_annotations = collect_data(
                fname,
                visited_funcs,
                visited_funcs_arguments,
                visited_funcs_retval,
                namespace,
            )
            import_param = imports
            if not insert_imports:
                import_param = set()
            args = (
                fname,
                type_annotations,
                import_param,
                overwrite,
                not_annotated,
                ignore_annotations,
                srcdir,
            )
            if use_multiprocessing:
                process = multiprocessing.Process(
                    target=process_file, args=args
                )
                processes.append(process)
                process.start()
            else:
                process_file(*args)
                progress.update(task1, advance=1)

        if use_multiprocessing:
            while processes:
                for (
                    p
                ) in (
                    processes
                ):  # multiprocessing.connection.wait(processes, timeout=None):
                    p.join()
                    processes.remove(p)
                    progress.update(task1, advance=1)


def should_update_file(
    t: FuncInfo,
    fname: str,
    prefix: str,
    namespace: Dict[str, Any],
) -> bool:
    try:
        s = make_type_signature(
            file_name=t.file_name,
            func_name=t.func_name,
            args=visited_funcs_arguments[t],
            retval=visited_funcs_retval[t],
            namespace=namespace,
            not_annotated=not_annotated,
            arg_types=arg_types,
            existing_annotations=existing_annotations,
        )
    except KeyError:
        return False
    return t not in existing_spec or s != existing_spec[t]


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
    "--srcdir",
    type=click.Path(exists=True, file_okay=False),
    default=os.getcwd(),
    help="Use this as the base for imports.",
)
@click.option(
    "--overwrite/--no-overwrite",
    help="Overwrite files with type information.",
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
    "--insert-imports",
    is_flag=True,
    help="Insert import statements for missing classes (MAY LEAD TO CIRCULAR IMPORTS).",
    default=False,
)
@click.option(
    "--generate-stubs",
    is_flag=True,
    help="Generate stub files (.pyi).",
    default=False,
)
@click.option(
    "--type-coverage-by-directory",
    type=click.Path(exists=True, file_okay=False),
    help="Report per-directory type annotation coverage for all Python files in a directory and its children.",
)
@click.option(
    "--type-coverage-by-file",
    type=click.Path(exists=True, file_okay=False),
    help="Report per-file type annotation coverage for all Python files in a directory or its children.",
)
@click.option(
    "--type-coverage-summary",
    type=click.Path(exists=True, file_okay=False),
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
def main(
    script: str,
    all_files: bool,
    include_files: str,
    module: bool,
    triple_dash: bool,
    args: List[str],
    type_coverage_by_directory: str,
    type_coverage_by_file: str,
    type_coverage_summary: str,
    verbose: bool,
    overwrite: bool,
    ignore_annotations: bool,
    insert_imports: bool,
    generate_stubs: bool,
    srcdir: str,
) -> None:
    """
    RightTyper efficiently generates types for your function
    arguments and return values.
    """
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

    initialize_globals(include_files, all_files, script, verbose)
    tool_args, script_args = split_args_at_triple_dash(args)
    setup_tool_id()
    register_monitoring_callbacks(
        enter_function,
        call_handler,
        exit_function,
        yield_function,
        ignore_annotations,
    )
    setup_timer(handle_clear_seen_funcs)
    execute_script_or_module(script, module, tool_args, script_args)
    reset_monitoring()
    post_process(
        overwrite=overwrite,
        ignore_annotations=ignore_annotations,
        insert_imports=insert_imports,
        generate_stubs=generate_stubs,
        srcdir=srcdir,
    )


def split_args_at_triple_dash(
    args: List[str],
) -> Tuple[List[str], List[str]]:
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
