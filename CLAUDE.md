# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RightTyper is a Python runtime type inference tool that automatically generates type annotations for function arguments and return values. It uses Python 3.12+ `sys.monitoring` for low-overhead (~25%) type observation during test runs, then transforms source files to add the inferred annotations.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .
pip install -e ".[tests]"  # With test dependencies

# Run tests
pytest                              # All tests
pytest tests/test_file.py           # Specific test file
pytest tests/test_file.py::test_name  # Single test
pytest -n0 -x tests/test_issues.py  # Serially, stopping at the first failure

# CLI usage
python3 -m righttyper run your_script.py [args]
python3 -m righttyper run -m pytest [pytest-args]
python3 -m righttyper process [options]
python3 -m righttyper coverage --type [directory]
```

**Do not run `make`.** The `Makefile` invokes `black -l 79 righttyper` while
`pyproject.toml` sets `line-length = 100`. They contradict each other, no CI job enforces
either, and running it reformats the entire package — which CONTRIBUTING.md explicitly asks
contributors not to do ("If you also reformat all the code, it will be hard for us to focus
on your change"). Match the style of the lines you touch. `black`, `ruff` and `pyright` are
declared in no extra, so `pip install -e ".[tests]"` does not install them.

**Use `-n0`, not `-p no:xdist`, to run tests serially.** `addopts = "-n auto"` means pytest
injects `-n` into every invocation, so disabling the plugin makes pytest reject its own flag.

## Architecture

### Core Pipeline

```
CLI (righttyper.py)
    ↓
┌───────────────┬─────────────────┐
│  run command  │ process command │
└───────┬───────┴────────┬────────┘
        ↓                ↓
┌───────────────────────────────────┐
│   Type Observation System         │
│   recorder.py → observations.py   │
│   type_id.py → typeinfo.py        │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│   Code Transformation             │
│   unified_transformer.py (CST)    │
│   generalize.py, annotation.py    │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│   Output                          │
│   Python files, .pyi stubs, JSON  │
└───────────────────────────────────┘
```

### Key Components

- **righttyper.py**: Main CLI with `run`, `process`, `coverage` commands using Click
- **righttyper_tool.py**: Sets up `sys.monitoring` hooks for function call/return observation
- **loader.py**: Meta-path finder and loader that AST-instruments modules as they are imported
- **recorder.py**: Records function invocations and their argument/return types
- **variable_capture.py**: Maps code objects to the variables assigned within them
- **observations.py**: Aggregates recorded type observations across multiple calls
- **type_id.py**: Converts runtime Python values to TypeInfo representations
- **unified_transformer.py**: libcst-based transformer that applies annotations to source
- **generalize.py**: Merges multiple observed types into unified type annotations (`lub()`)
- **typemap.py**: Maps type objects to a canonical name for the type, where one exists
- **typeshed.py**: Reads signatures out of typeshed stubs
- **type_transformers.py**: Filters and simplifies types (mock resolution, test exclusion, depth limiting)

### Type Recording Flow

1. `righttyper_tool.py` installs monitoring hooks via `sys.monitoring`
2. On each function call/return, `recorder.py` captures argument values and return value
3. `type_id.py` converts runtime values to TypeInfo structures
4. `observations.py` stores observations in a probabilistic data structure
5. Two independent sampling mechanisms limit the cost of the above:
   - **Call sampling**: past `--poisson-warmup-samples` (default 5), the return handler
     returns `sys.monitoring.DISABLE` for that code object. A daemon `threading.Timer`
     re-arms all previously-seen code at exponentially distributed intervals
     (`--poisson-rate`, default 2 windows/sec). `--no-call-sampling` disables this, and
     only this — it records every call and is much slower than the published overhead
     figure, which is measured with sampling on.
   - **Container sampling**: large containers are sampled rather than walked in full,
     stopping once the observed type distribution stabilizes (`--container-*` options).

### Code Transformation Flow

1. `righttyper_process.py` loads collected observations
2. `generalize.py` merges observations into canonical types
3. `type_transformers.py` applies filtering (mock resolution, depth limits)
4. `unified_transformer.py` uses libcst to rewrite source files with annotations
5. Output: modified .py files, .pyi stubs, or JSON

### Failure Handling and Logs

The post-run phase — generalization, type transformation, file writing — is wrapped in a
bare `except:` that logs and returns, re-raising only under `--allow-runtime-exceptions`.
A failure there therefore produces **exit 0, no console output and no annotations**, which
is indistinguishable from a run that found nothing. The traceback goes to `righttyper.log`,
which `logger.py` opens in the working directory on every run, successful or not.

When debugging a run that "did nothing", read `righttyper.log` first and re-run with
`--allow-runtime-exceptions`.

`sys.monitoring`'s `CALL` and `PY_UNWIND` events are registered process-globally via
`set_events` (not `set_local_events`), so the handlers fire for every call in the process.
`--exclude-files` filters what is *recorded*, not what is *observed*, and cannot suppress a
crash originating in an excluded file.

### Destructive Defaults

`overwrite` and `output_files` both default to **True** (`options.py`), so a bare `run`
rewrites source files in place, leaving `.py.bak` originals beside them. `.bak` is gitignored
in many projects, so `git status` can look clean while the tree has been rewritten. Use
`--no-output-files --no-overwrite` for a read-only run.

### Configuration

- **pyproject.toml**: Black line-length=100, isort with black profile — note the `Makefile`
  disagrees, passing `black -l 79`; nothing in CI enforces either
- **Python**: Requires 3.12-3.14
- **Dependencies**: libcst (CST parsing), click (CLI), typeshed_client, dill (serialization)

### Test Structure

- `test_integration.py`: End-to-end tests running full pipeline
- `test_transformer.py`: Unit tests for CST transformation logic
- `test_typing.py`: Type handling and inference tests
- `test_generalize.py`: Type generalization algorithm tests
- Pytest markers: `dont_run_mypy`, `mypy_args` for controlling mypy validation in tests
- Use `--no-mypy` flag when running tests without mypy installed

### CLI Option Organization

- Use `click_option_group` (`optgroup`) to group related options in help output
- Create decorator functions like `add_output_options()` and `add_advanced_options()` to apply option groups
- Options in groups are applied via decorators: `@add_advanced_options(group="Advanced options")`
- Rarely-used options go in "Advanced options" group instead of being hidden

### Wrapped Function Type Propagation

RightTyper handles decorators where the wrapped function never executes (e.g., JIT compilers, `functools.wraps`):

1. **Detection**: In `recorder.py`, `_record_wrapped_function_types()` detects wrapped functions via:
   - `__call__` methods on objects with `__wrapped__` attribute
   - Regular functions with `__wrapped__` attribute
   - Wrapper functions created by `functools.wraps`

2. **Pending Traces Pattern**: Since wrapped functions don't execute, we can't observe their return type directly:
   - Store pending trace at wrapper invocation: `_pending_wrapped_traces[(wrapper_code, frame_id)] = (wrapped_code, arg_types)`
   - Complete trace when wrapper returns: use wrapper's return type for the wrapped function
   - Clean up on exception: discard pending trace if wrapper raises

3. **Configurable via `--infer-wrapped-return-type`**:
   - Default (enabled): infer return type from wrapper's actual return value
   - Disabled: use `None` as placeholder return type
