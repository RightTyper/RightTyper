# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RightTyper is a Python runtime type inference tool that automatically generates type annotations for function arguments and return values. It uses Python 3.12+ `sys.monitoring` for low-overhead (~25%) type observation during test runs, then transforms source files to add the inferred annotations.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .
pip install -e ".[tests]"  # With test dependencies

# Linting and formatting (via Makefile)
make all       # Run black, ruff, pyright
make black     # Format with black
make ruff      # Lint with ruff
make pyright   # Type check with pyright

# Run tests
pytest                          # All tests
pytest tests/test_file.py       # Specific test file
pytest tests/test_file.py::test_name  # Single test

# CLI usage
python3 -m righttyper run your_script.py [args]
python3 -m righttyper run -m pytest [pytest-args]
python3 -m righttyper process [options]
python3 -m righttyper coverage --type [directory]
```

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

- **righttyper.py** (1041 lines): Main CLI with `run`, `process`, `coverage` commands using Click
- **righttyper_tool.py**: Sets up `sys.monitoring` hooks for function call/return observation
- **recorder.py**: Records function invocations and their argument/return types
- **observations.py**: Aggregates recorded type observations across multiple calls
- **type_id.py** (771 lines): Converts runtime Python values to TypeInfo representations
- **unified_transformer.py** (1311 lines): libcst-based transformer that applies annotations to source
- **generalize.py**: Merges multiple observed types into unified type annotations
- **type_transformers.py**: Filters and simplifies types (mock resolution, test exclusion, depth limiting)

### Type Recording Flow

1. `righttyper_tool.py` installs monitoring hooks via `sys.monitoring`
2. On each function call/return, `recorder.py` captures argument values and return value
3. `type_id.py` converts runtime values to TypeInfo structures
4. `observations.py` stores observations in a probabilistic data structure
5. Sampling stops when type distribution stabilizes (configurable thresholds)

### Code Transformation Flow

1. `righttyper_process.py` loads collected observations
2. `generalize.py` merges observations into canonical types
3. `type_transformers.py` applies filtering (mock resolution, depth limits)
4. `unified_transformer.py` uses libcst to rewrite source files with annotations
5. Output: modified .py files, .pyi stubs, or JSON

### Configuration

- **pyproject.toml**: Black line-length=100, isort with black profile
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
   - Case 1: `__call__` methods on objects with `__wrapped__` attribute
   - Case 2: Regular functions with `__wrapped__` attribute
   - Case 3: Wrapper functions created by `functools.wraps` (closure-based detection)
   - Case 4: Tracing JIT pattern where function and representative args are passed together

2. **Pending Traces Pattern**: Since wrapped functions don't execute, we can't observe their return type directly:
   - Store pending trace at wrapper invocation: `_pending_wrapped_traces[(wrapper_code, frame_id)] = (wrapped_code, arg_types, trace_count)`
   - Complete trace when wrapper returns: use wrapper's return type for the wrapped function
   - Skip wrapped trace if the wrapped function was directly observed during wrapper execution (avoids duplicate/conflicting traces)
   - Clean up on exception: discard pending trace if wrapper raises

3. **Configurable via `--infer-wrapped-return-type`**:
   - Default (enabled): infer return type from wrapper's actual return value
   - Disabled: use `None` as placeholder return type

4. **Classmethod and Staticmethod Support**: The `unwrap()` function in `type_id.py` handles descriptor unwrapping:
   - Follows `__func__` attribute on `classmethod` and `staticmethod` descriptors
   - Then follows `__wrapped__` chain to find the original function
   - Example: `@classmethod @decorator def foo(cls, x)` → unwrap finds original `foo`

5. **Wrapper Argument Capture**: When capturing arguments to propagate to wrapped functions:
   - Captures both regular positional parameters (e.g., `cls` in `def wrapper(cls, *args, **kwargs)`)
   - Plus `*args` and `**kwargs` values
   - This ensures classmethod wrappers correctly propagate the `cls` argument

6. **Avoiding Decorator vs Wrapper Confusion**: Case 3 detection (closure-based `func` lookup) checks that `func` is a closure variable, not a function argument. This prevents false triggering when the decorator itself is called (e.g., `decorator(foo)`) vs when the wrapper is called.

7. **Tracing JIT Pattern (Case 4)**: Handles tracing-based JIT decorators where function and representative arguments are passed together:
   - Pattern: `trace(func, *args)` where `func` is the function to type and `*args` are representative values
   - Detection: function has a callable argument named `func`/`f`/`fn`/`function`/`callable` AND has `*args` with values
   - Only `*args` values are propagated (not `cls` or `func` parameters which are decorator machinery)
   - Example: `TraceKernel.trace(add, 3, 4)` infers types `add(a: int, b: int) -> ...`
   - Reference: https://eli.thegreenplace.net/2025/decorator-jits-python-as-a-dsl/
