import concurrent.futures
import importlib.metadata
import importlib.util
import inspect
import functools
import logging
import os
import runpy
import sys
import dill as pickle
import datetime
import json
import subprocess
import re
import time

from dataclasses import asdict
from pathlib import Path
from types import CodeType, FrameType, GeneratorType, AsyncGeneratorType
from typing import Any, TextIO

import click
from click_option_group import optgroup

from righttyper.righttyper_process import (
    process_file,
    CodeChanges
)
from righttyper.righttyper_tool import (
    TOOL_ID,
    TOOL_NAME,
    setup_monitoring,
    shutdown_monitoring,
    stop_events,
    restart_events,
    enabled_code
)
import righttyper.loader as loader
from righttyper.righttyper_utils import detected_test_modules
from righttyper.typeinfo import TypeInfo
from righttyper.righttyper_types import CodeId, Filename, FunctionName
from righttyper.annotation import FuncAnnotation, ModuleVars
from righttyper.observations import Observations
from righttyper.recorder import ObservationsRecorder
from righttyper.options import run_options, output_options
from righttyper.logger import logger
from righttyper.atomic import AtomicCounter


PKL_FILE_NAME = TOOL_NAME+"-{N}.rt"
PKL_FILE_VERSION = 6


rec = ObservationsRecorder()
instrumentation_counter = AtomicCounter()


def is_instrumentation(disable_retval=sys.monitoring.DISABLE):
    def decorator(f):
        """Decorator that marks a function as being instrumentation."""
        def wrapper(*args, **kwargs):
            try:
                if args[0] not in enabled_code:
                    return disable_retval

                instrumentation_counter.inc()
                return f(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except:
                logger.error("exception in instrumentation", exc_info=True)
                if run_options.allow_runtime_exceptions: raise

        return wrapper
    return decorator


@is_instrumentation()
def send_handler(code: CodeType, frame: FrameType, arg0: Any) -> None:
    rec.record_send(code, frame, arg0)


def wrap_send(obj: Any) -> Any:
    if (
        (self := getattr(obj, "__self__", None)) and
        isinstance(self, (GeneratorType, AsyncGeneratorType))
    ):
        if isinstance(self, GeneratorType):
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                # generator.send takes exactly one argument
                send_handler(self.gi_code, self.gi_frame, args[0])
                return obj(*args, **kwargs)

            return wrapper
        else:
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                # generator.asend takes exactly one argument
                send_handler(self.ag_code, self.ag_frame, args[0])
                return obj(*args, **kwargs)

            return wrapper

    return obj


@is_instrumentation()
def start_handler(code: CodeType, offset: int) -> Any:
    """
    Process the function entry point, perform monitoring related operations,
    and manage the profiling of function execution.
    """
    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    if frame:
        rec.record_start(code, frame, inspect.getargvalues(frame))
        del frame

    return None


@is_instrumentation()
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
    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    if frame:
        rec.record_yield(code, frame, yield_value)
        del frame

    return None


@is_instrumentation()
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
    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    found = frame and rec.record_return(code, frame, return_value)
    del frame

    if (
        found
        and run_options.sampling
        and not (
            (no_sampling_for := run_options.no_sampling_for_re)
            and no_sampling_for.search(code.co_qualname)
        )
    ):
        # Poisson sampling: disable after each sample once past warmup
        if rec.past_warmup(code):
            stop_events(code)
            rec.clear_pending(code)
            return sys.monitoring.DISABLE

    return None


@is_instrumentation(disable_retval=None)
def unwind_handler(
    code: CodeType,
    instruction_offset: int,
    exception: BaseException,
) -> Any:
    frame = inspect.currentframe()
    while frame and frame.f_code is not code:
        frame = frame.f_back

    found = frame and rec.record_no_return(code, frame)
    del frame

    if (
        found
        and run_options.sampling
        and not (
            (no_sampling_for := run_options.no_sampling_for_re)
            and no_sampling_for.search(code.co_qualname)
        )
    ):
        # Poisson sampling: disable until next timer (once past warmup)
        if rec.past_warmup(code):
            stop_events(code)
            rec.clear_pending(code)

    return None  # PY_UNWIND can't be disabled


import random
import threading

# Poisson-timed sampling state
capture_timer: threading.Timer | None = None
hist_capture_times: list[float] = []

def schedule_next_capture() -> None:
    """Schedule the next monitoring window at a random future time."""
    global capture_timer

    # Exponential inter-arrival time (Poisson process)
    delay = random.expovariate(run_options.poisson_sample_rate)
    capture_timer = threading.Timer(delay, begin_capture)
    capture_timer.daemon = True
    capture_timer.start()

def begin_capture() -> None:
    """Enable monitoring briefly to capture samples."""
    if run_options.save_profiling:
        hist_capture_times.append(time.perf_counter())

    restart_events()  # Re-enable monitoring for all previously-seen code
    schedule_next_capture()

def stop_capture() -> None:
    """Cancel any pending capture timer."""
    global capture_timer
    if capture_timer:
        capture_timer.cancel()
        capture_timer = None


main_globals: dict[str, Any] = dict()

def execute_script_or_module(
    script: str,
    is_module: bool,
    args: list[str],
) -> None:
    """Executes the script or module, returning the __main__ module's globals."""

    global main_globals

    try:
        sys.argv = [script, *args]
        if is_module:
            with loader.ImportManager(replace_dict=run_options.replace_dict):
                main_globals = runpy.run_module(
                    script,
                    run_name="__main__",
                    alter_sys=True,
                )
        else:
            with loader.ImportManager(replace_dict=run_options.replace_dict):
                main_globals = runpy.run_path(script, run_name="__main__")

    except BaseException as e:
        tb = e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_globals.get("__name__", None) == "__main__":
                main_globals = dict(tb.tb_frame.f_globals)
                break
            tb = tb.tb_next

        if not isinstance(e, SystemExit) or e.code not in (None, 0):
            raise


def output_changes(
    code_changes: list[CodeChanges],
    file: TextIO = sys.stdout,
) -> None:
    import difflib

    for filename, changes in sorted(code_changes):
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


def emit_json(
    files: list[tuple[Filename, str]],
    type_annotations: dict[CodeId, FuncAnnotation],
    module_vars: dict[Filename, ModuleVars],
    code_changes: list[CodeChanges]
) -> dict[str, Any]:

    file2module = {file: module for file, module in files}
    file_func2sigs = {
        (file, funcname): (old_sig, new_sig)
        for file, changes in code_changes
        for funcname, old_sig, new_sig in changes
    }

    data: dict[str, Any] = {
        'meta': {
            'software': TOOL_NAME,
            'version': importlib.metadata.version(TOOL_NAME),
            'timestamp': datetime.datetime.now().isoformat(),
        },
        'files': {
            filename: {
                'module': file2module.get(filename),
                'functions': {},
            }
            for filename in sorted(
                {
                    funcid.file_name
                    for funcid in type_annotations
                }
                | set(module_vars)
            )
        }
    }

    # fill in functions
    for funcid, ann in type_annotations.items():
        if funcid.func_name.endswith(">"):  # <genexpr> and such
            continue

        func_json_name = funcid.func_name.replace(".<locals>.", ".")
        file_functions: dict[str, Any] = data['files'][funcid.file_name]['functions']
        if func_json_name in file_functions:
            continue  # TODO handle functions with the same name

        # varargs/kwargs arguments are implicitly tuples and dicts in annotations,
        # but lacking the context given by their names prefixed by * or **, in JSON
        # we want to be explicit about those types.
        def argtype(name: str, t: TypeInfo) -> str:
            if name == ann.varargs:
                return str(TypeInfo.from_type(tuple, module='', args=(t, ...)))
            if name == ann.kwargs:
                return str(TypeInfo.from_type(dict, module='', args=(TypeInfo.from_type(str, module=''), t)))
            return str(t)

        func_entry = file_functions[func_json_name] = {
            'args': {
                a[0]: argtype(*a).replace(".<locals>.", ".")
                for a in ann.args
            },
            'retval': str(ann.retval).replace(".<locals>.", "."),
            'vars': {
                v[0]: str(v[1]).replace(".<locals>.", ".")
                for v in ann.variables
            }
        }

        if changes := file_func2sigs.get((funcid.file_name, funcid.func_name)):
            func_entry['old_sig'] = changes[0]
            func_entry['new_sig'] = changes[1]

    # fill in module vars
    for filename, mv in module_vars.items():
        data['files'][filename]['vars'] = {
            k: str(v).replace(".<locals>.", ".")
            for k, v in mv.variables
        }

    return data


def process_obs(obs: Observations):
    files = list(obs.source_to_module_name.items())
    type_annotations, module_vars = obs.collect_annotations()
    logger.debug(f"generated {len(type_annotations)} annotation(s)")

    code_changes: list[CodeChanges] = process_files(files, type_annotations, module_vars)

    if output_options.json_output:
        data = emit_json(files, type_annotations, module_vars, code_changes)
        with open(f"{TOOL_NAME}.json", "w") as f:
            json.dump(data, f, indent=2)

    else:
        with open(f"{TOOL_NAME}.out", "w+") as f:
            output_changes(code_changes, f)


def process_file_wrapper(args) -> CodeChanges:
    return process_file(*args)


def process_files(
    files: list[tuple[Filename, str]],
    type_annotations: dict[CodeId, FuncAnnotation],
    module_vars: dict[Filename, ModuleVars]
) -> list[CodeChanges]:
    if not files:
        return []

    args_gen = (
        (
            file[0],    # path
            file[1],    # module_name
            type_annotations,
            module_vars.get(file[0]),
            output_options
        )
        for file in files
    )

    if output_options.use_multiprocessing:
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

    code_changes = []

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
            code_changes.append(result)
            progress.update(task1, advance=1)
            progress.refresh()

    return code_changes


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

UNMATCHED_BRACKET = re.compile(r"\[[^]]*$")

def validate_fnmatch(ctx, param, value):
    if value:
        for v in value:
            if not v.strip():
                raise click.BadParameter("Pattern may not be empty.")
            if UNMATCHED_BRACKET.search(v):
                raise click.BadParameter(f"Unmatched '[' in pattern: {value!r}")

        # make patterns absolute, as co_filename strings are also absolute
        return tuple(
            str(Path(v).absolute())
            for v in value
        )
    return value


class HelpfulGroup(click.Group):
    def __init__(self, *args, extra_help_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_help_subcommands = tuple(extra_help_subcommands or ())

    def get_help(self, ctx):
        help_text = [super().get_help(ctx)]

        for name in ('run',):
            cmd = self.get_command(ctx, name)
            if cmd is not None:
                subctx = cmd.make_context(name, [], parent=ctx, resilient_parsing=True)
                help_text.append(f"\n\n---- Help for '{name}': ----\n{cmd.get_help(subctx)}")

        return "".join(help_text)


@click.group(
    cls=HelpfulGroup,
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


def add_output_options(group=None):
    """Decorates a click command, adding our common output options."""

    def dec(func):
        base = optgroup if group else click

        for opt in reversed([
            *(
                (optgroup.group(group),) if group else ()
            ),
            base.option(
                "--overwrite/--no-overwrite",
                help="""Overwrite ".py" files with type information. If disabled, ".py.typed" files are written instead. The original files are saved as ".py.bak".""",
                default=output_options.overwrite,
            ),
            base.option(
                "--output-files/--no-output-files",
                help=f"Output annotated files (possibly overwriting, if specified).  If disabled, the annotations are only written to {TOOL_NAME}.out.",
                default=output_options.output_files,
            ),
            base.option(
                "--ignore-annotations",
                is_flag=True,
                help="Ignore existing annotations and overwrite with type information.",
                default=False,
            ),
            base.option(
                "--only-update-annotations",
                is_flag=True,
                default=False,
                help="Overwrite existing annotations but never add new ones.",
            ),
            base.option(
                "--generate-stubs",
                is_flag=True,
                help="Generate stub files (.pyi).",
                default=False,
            ),
            base.option(
                "--json-output",
                default=output_options.json_output,
                is_flag=True,
                help=f"Output inferences in JSON, instead of writing {TOOL_NAME}.out."
            ),
            base.option(
                "--use-multiprocessing/--no-use-multiprocessing",
                default=True,
                help="Whether to use multiprocessing.",
            ),
            base.option(
                "--type-depth-limit",
                default="none",
                callback=lambda ctx, param, value: parse_none_or_ge_zero(value),
                show_default=True,
                metavar="[INTEGER|none]",
                help="Maximum depth (types within types) for generic types; 'none' to disable.",
            ),
            base.option(
                "--python-version",
                type=click.Choice(["3.9", "3.10", "3.11", "3.12", "3.13"]),
                default="3.12",
                callback=lambda ctx, param, value: tuple(int(n) for n in value.split('.')),
                help="Python version for which to emit annotations.",
            ),
            base.option(
                "--use-top-pct",
                type=click.IntRange(1, 100),
                default=output_options.use_top_pct,
                metavar="PCT",
                help="Only use the PCT% most common call traces.",
            ),
            base.option(
                "--use-typing-never/--no-use-typing-never",
                default=output_options.use_typing_never,
                help="""Whether to emit "typing.Never" (for Python versions that support it).""",
            ),
            base.option(
                "--simplify-types/--no-simplify-types",
                default=output_options.simplify_types,
                help="Whether to attempt to simplify types, such as int|bool|float -> float. or Generator[X, None, None] -> Iterator[X]",
            ),
            base.option(
                "--exclude-test-types/--no-exclude-test-types",
                is_flag=True,
                default=output_options.exclude_test_types,
                help="""Whether to exclude or replace with "typing.Any" types defined in test modules."""
            ),
            base.option(
                "--always-quote-annotations/--no-always-quote-annotations",
                is_flag=True,
                default=output_options.always_quote_annotations,
                help="""Place all annotations in quotes. This is normally not necessary, but can help avoid undefined symbol errors."""
            ),
        ]):
            func = opt(func)
        return func
    return dec


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
    "--exclude-files",
    metavar="GLOB",
    type=str,
    multiple=True,
    callback=validate_fnmatch,
    help="Exclude the given files (using fnmatch). Can be passed multiple times.",
)
@click.option(
    "--exclude-test-files/--no-exclude-test-files",
    default=run_options.exclude_test_files,
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
    "--poisson-rate",
    type=click.FloatRange(0.1, None),
    default=run_options.poisson_sample_rate,
    help="Expected sample captures per second (Poisson process rate).",
)
@click.option(
    "--sampling/--no-sampling",
    default=run_options.sampling,
    help=f"Whether to sample calls or to use every one.",
)
@click.option(
    "--no-sampling-for",
    metavar="REGEX",
    type=str,
    multiple=True,
    callback=validate_regexes,
    default=run_options.no_sampling_for,
    help=f"Rather than sample, record every invocation of any functions matching the given regular expression. Can be passed multiple times.",
)
@click.option(
    "--replace-dict/--no-replace-dict",
    is_flag=True,
    help="Whether to replace 'dict' to enable efficient, statistically correct samples."
)
@click.option(
    "--container-small-threshold",
    type=click.IntRange(1, None),
    default=run_options.container_small_threshold,
    help="Containers at or below this size are fully scanned instead of sampled.",
)
@click.option(
    "--container-max-samples",
    type=click.IntRange(1, None),
    default=run_options.container_max_samples,
    help="Maximum number of entries to sample for a container.",
)
@click.option(
    "--container-type-threshold",
    type=click.FloatRange(0.01, None),
    default=run_options.container_type_threshold,
    help="Stop sampling a container if the estimated likelihood of finding a new type falls below this threshold.",
)
@click.option(
    "--container-sample-range",
    default="1000",
    callback=lambda ctx, param, value: parse_none_or_ge_zero(value),
    show_default=True,
    metavar="[INTEGER|none]",
    help="Largest index from which to sample in a container when direct access isn't available; 'none' means unlimited.",
)
@click.option(
    "--container-min-samples",
    type=click.IntRange(1, None),
    default=run_options.container_min_samples,
    help="Minimum samples before checking Good-Turing stopping criterion.",
)
@click.option(
    "--container-check-probability",
    type=click.FloatRange(0.0, 1.0),
    default=run_options.container_check_probability,
    help="Probability of spot-checking a container for new types.",
)
@click.option(
    "--save-profiling",
    is_flag=True,
    hidden=True,
    help=f"""Save record of self-profiling results in "{TOOL_NAME}-profiling.json", under the given name."""
)
@click.option(
    "--resolve-mocks/--no-resolve-mocks",
    is_flag=True,
    default=run_options.resolve_mocks,
    help="Whether to attempt to resolve test types, such as mocks, to non-test types."
)
@click.option(
    "--test-modules",
    multiple=True,
    default=run_options.test_modules,
    callback=validate_module_names,
    metavar="MODULE",
    help="""Additional modules (besides those detected) whose types are subject to mock resolution or test type exclusion, if enabled. Matches submodules as well. Can be passed multiple times."""
)
@click.option(
    "--adjust-type-names/--no-adjust-type-names",
    default=run_options.adjust_type_names,
    help="Whether to look for a canonical name for types, rather than use the module and name where they are defined.",
)
@click.option(
    "--variables/--no-variables",
    default=run_options.variables,
    help="Whether to (observe and) annotate variables.",
)
@click.option(
    "--only-collect",
    default=False,
    is_flag=True,
    help=f"Rather than immediately process collect data, save it to \"{PKL_FILE_NAME.format(N='N')}\"." +\
          " You can later process using RightTyper's \"process\" command."
)
@click.option(
    "--allow-runtime-exceptions/--no-allow-runtime-exceptions",
    is_flag=True,
    default=run_options.allow_runtime_exceptions,
    hidden=True,
)
@click.option(
    "--generalize-tuples",
    metavar="N",
    type=click.IntRange(0, None),
    default=run_options.generalize_tuples,
    help="Generalize homogenous fixed-length tuples to tuple[T, ...] if length ≥ N.  N=0 disables generalization."
)
@click.option(
    "--debug",
    is_flag=True,
    help="Include diagnostic information in log file.",
)
@add_output_options(group="Output options")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def run(
    script: str,
    module: str,
    root: str,
    args: list[str],
    poisson_rate: float,
    only_collect: bool,
    debug: bool,
    **kwargs,
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

    if root:
        run_options.script_dir = os.path.realpath(root)
    elif module:
        run_options.script_dir = os.getcwd()
    else:
        run_options.script_dir = os.path.dirname(os.path.realpath(script))

    if not run_options.exclude_test_files:
        raise click.UsageError("Typing test files is temporarily disabled.")

    run_options.poisson_sample_rate = poisson_rate
    run_options.process_args(kwargs)
    output_options.process_args(kwargs)

    if run_options.infer_shapes:
        packages_needed = {"jaxtyping"}
        packages_found = {
            pkg_name
            for pkg_name in packages_needed
            if importlib.util.find_spec(pkg_name) is not None
        }

        if (missing := packages_needed - packages_found):
            print("The following package(s) need to be installed for '--infer-shapes':")
            for package in missing:
                print(f" * {package}")
            sys.exit(1)

    pytest_plugins = os.environ.get("PYTEST_PLUGINS")
    pytest_plugins = (pytest_plugins + "," if pytest_plugins else "") + "righttyper.pytest"
    os.environ["PYTEST_PLUGINS"] = pytest_plugins

    setup_monitoring(
        start_handler,
        yield_handler,
        return_handler,
        unwind_handler,
    )

    if run_options.sampling:
        schedule_next_capture()

    try:
        execute_script_or_module(script, is_module=bool(module), args=args)
    finally:
        rec.try_close_generators()
        shutdown_monitoring()
        stop_capture()

        try:
            obs = rec.finish_recording(main_globals)

            logger.debug(f"observed {len(obs.source_to_module_name)} file(s)")

            if logger.level == logging.DEBUG:
                for m in detected_test_modules:
                    logger.debug(f"test module: {m}")

            if only_collect:
                from righttyper.type_transformers import MakePickleableT
                obs.transform_types(MakePickleableT())

                collected = {
                    'file_version': PKL_FILE_VERSION,
                    'software': TOOL_NAME,
                    'version': importlib.metadata.version(TOOL_NAME),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'run_options': run_options,
                    'script': Path(script).resolve(),
                    'observations': obs
                }

                index = 1
                while True:
                    filename = Path(PKL_FILE_NAME.format(N=index))
                    try:
                        with filename.open("xb") as pklf:
                            pickle.dump(collected, pklf)
                            break

                    except FileExistsError:
                        index += 1

                print(f"Collected types saved to {filename}.")
            else:
#                from righttyper.type_transformers import MakePickleableT, LoadTypeObjT
#                obs.transform_types(MakePickleableT())
#                obs.transform_types(LoadTypeObjT())
                process_obs(obs)
        except:
            logger.error("exception after target execution", exc_info=True)
            if run_options.allow_runtime_exceptions: raise

        end_time = time.perf_counter()
        logger.info(f"Finished in {end_time-start_time:.0f}s")

        if run_options.save_profiling:
            try:
                with open(f"{TOOL_NAME}-profiling.json", "r") as pf:
                    data = json.load(pf)
            except FileNotFoundError:
                data = []

            data.append({
                    'command': subprocess.list2cmdline(sys.orig_argv),
                    'start_time': start_time,
                    'end_time': end_time,
                    'elapsed': end_time - start_time,
                    'poisson_rate': run_options.poisson_sample_rate,
                    'capture_times': hist_capture_times,
                }
            )

            with open(f"{TOOL_NAME}-profiling.json", "w") as pf:
                json.dump(data, pf, indent=2)


@cli.command()
@add_output_options()
def process(**kwargs):
    """Processes type information collected with the 'run' command."""
    output_options.process_args(kwargs)

    obs_list = []
    script = None
    for filename in Path('.').glob(PKL_FILE_NAME.format(N='*')):
        with filename.open("rb") as f:
            pkl = pickle.load(f)

            if pkl.get('file_version') != PKL_FILE_VERSION:
                print(f"Error: Unsupported version in {filename}: " +\
                      f"{pkl.get('file_version')}, expected {PKL_FILE_VERSION}")
                sys.exit(1)

            if script and pkl['script'] != script:
                print(f"Error: {filename} was collected for {pkl['script']}, but previous file(s) were for {script}")
                sys.exit(1)

            # TODO check for compatible options?
            obs_list.append(pkl['observations'])

    if not obs_list:
        print("Error: No files found")
        sys.exit(1)

    # Copy run options, but be careful not to replace instance, as there may be
    # multiple references to it (e.g., through "from .options import ...")
    # TODO check, but this shouldn't be necessary, as we only use run options
    global options
    for key, value in asdict(pkl['run_options']).items():
        setattr(run_options, key, value)

    obs = obs_list[0]
    for obs2 in obs_list[1:]:
        obs.merge_observations(obs2)

    from righttyper.type_transformers import LoadTypeObjT
    obs.transform_types(LoadTypeObjT())
    process_obs(obs)


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
