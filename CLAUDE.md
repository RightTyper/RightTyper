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

### Lazy Imports for Runtime Type Hints (`--no-type-checking`)

When code calls `typing.get_type_hints()` at runtime, it needs the imported modules to be available in the function's namespace. However, placing imports at the top level can cause circular import errors in complex codebases.

The `--no-type-checking` option solves this using **lazy module proxies** (not `__getattr__`):

1. **Problem**:
   - Standard behavior adds imports under `TYPE_CHECKING` guard
   - `typing.get_type_hints()` fails with `NameError` because imports aren't available at runtime
   - Moving imports to top level causes circular import errors
   - **Key insight**: Module-level `__getattr__` (PEP 562) doesn't work because `get_type_hints()` uses direct dict lookup on `globals()`, not attribute access

2. **Solution** (unified_transformer.py):
   - Keep `from __future__ import annotations` (annotations remain strings)
   - Keep `TYPE_CHECKING` imports (for static analysis tools like mypy)
   - Add a `_LazyModule` proxy class that defers imports until first attribute access
   - Create proxy instances in `globals()` for each top-level module needed

3. **Generated Code Pattern**:
   ```python
   from __future__ import annotations
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       import neuronxcc.some.nested.module

   # ... functions with annotations ...

   class _LazyModule:
       """Proxy that lazily imports a module for typing.get_type_hints() support."""
       __slots__ = ("_rt_top_name", "_rt_full_path", "_rt_mod")
       def __init__(self, top_name, full_path):
           object.__setattr__(self, "_rt_top_name", top_name)
           object.__setattr__(self, "_rt_full_path", full_path)
           object.__setattr__(self, "_rt_mod", None)
       def _rt_load(self):
           mod = object.__getattribute__(self, "_rt_mod")
           if mod is None:
               import importlib, sys
               full_path = object.__getattribute__(self, "_rt_full_path")
               top_name = object.__getattribute__(self, "_rt_top_name")
               importlib.import_module(full_path)  # Populates intermediate modules
               mod = sys.modules[top_name]
               object.__setattr__(self, "_rt_mod", mod)
           return mod
       def __getattr__(self, name):
           return getattr(self._rt_load(), name)

   neuronxcc = _LazyModule('neuronxcc', 'neuronxcc.some.nested.module')
   ```

4. **Why This Works**:
   - At module load time, only proxy objects are created (no actual imports)
   - `TYPE_CHECKING` imports satisfy static type checkers
   - When `get_type_hints()` evaluates annotation string `neuronxcc.some.Foo`:
     1. It looks up `neuronxcc` in `globals()` → finds the proxy
     2. Accesses `.some` on the proxy → triggers `__getattr__` → loads full module path
     3. After import, `neuronxcc.some.Foo` is now fully resolved
   - By this time, all modules are loaded, so circular imports succeed

5. **Usage**:
   ```bash
   python3 -m righttyper run --no-type-checking -m pytest tests/
   ```

6. **Key Files**:
   - `unified_transformer.py`: Contains the lazy proxy generation logic in `leave_Module()`
   - `options.py`: Defines `no_type_checking` option
   - `righttyper.py`: Exposes `--no-type-checking` CLI flag

## Testing RightTyper on neuronxcc/starfish

### Background

The neuronxcc compiler uses `typing.get_type_hints()` at runtime (in `sema.py`, `TraceKernel.py`, `datamodel.py`) to inspect function signatures. When RightTyper adds annotations with TYPE_CHECKING imports, these fail with `NameError` at runtime.

### Test Plan for Starfish Tests

1. **Preparation**:
   ```bash
   cd /home/emerydb/workspace/KaenaCompiler/python-3.12

   # Create backup of original neuronxcc
   tar -cf neuronxcc.orig.tar neuronxcc/

   # Or restore from backup if needed
   rm -rf /tmp/neuronxcc-restore && mkdir -p /tmp/neuronxcc-restore
   tar -xf neuronxcc.orig.tar -C /tmp/neuronxcc-restore
   rsync -av --delete /tmp/neuronxcc-restore/neuronxcc/ neuronxcc/
   ```

2. **Run RightTyper on Starfish Tests**:
   ```bash
   # Run with --no-type-checking to use lazy imports
   NEURON_SKIP_VERIFY=1 python3 -m righttyper run \
       --no-type-checking \
       --overwrite \
       -m pytest neuronxcc/starfish/test/test*.py -x -q
   ```

3. **Verify Tests Pass After Annotation**:
   ```bash
   # Run tests again WITHOUT RightTyper to verify annotations work
   python3 -m pytest neuronxcc/starfish/test/test*.py -x -q
   ```

4. **Check for Regressions**:
   - Import errors (circular imports)
   - NameError from get_type_hints()
   - Runtime type evaluation failures

### Key Test Files in Starfish

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `test_bir_codegen.py` | 54 | BIR code generation |
| `test_insert_offloaded_transposes.py` | 22 | Transpose insertion |
| `test_fused_kernels_numerical.py` | ~100+ | Kernel fusion |
| `test_nki.py` | NKI tests | NKI compiler (may have special requirements) |

### Test Results (2026-01-06)

**Summary**: The lazy proxy implementation successfully fixes `NameError` issues with `typing.get_type_hints()`. However, the neuronxcc NKI sema code has a separate limitation that doesn't handle union types (`str|None`) in annotations.

1. **RightTyper run with `--no-type-checking`**:
   - Result: 6061 passed, 84 skipped, 4 failed (pre-existing test bugs)
   - No `NameError` from `get_type_hints()` during test execution

2. **Tests after annotation (without RightTyper)**:
   - Result: Many failures due to NKI sema union type limitation
   - Error: `AssertionError: Unexpected annotation! The first non-type hint is str | None`
   - Cause: `sema.py:704` expects all type hints to be `type` objects, but union types like `str|None` return `types.UnionType`

3. **Root Cause Analysis**:
   - The lazy proxy fix **works correctly** for `get_type_hints()` - no `NameError`
   - The issue is that NKI's type validation code (`sema.check_param_type`) doesn't handle union types
   - This is a **neuronxcc limitation**, not a RightTyper bug

4. **Recommendation for neuronxcc**:
   - Either avoid annotating NKI-decorated functions with union types
   - Or update `sema.py` to handle `types.UnionType` and `typing.Union` in addition to plain `type` objects

### Common Issues and Solutions

1. **Circular Import Errors**:
   - Symptom: `ImportError: cannot import name 'X' from partially initialized module`
   - Cause: Imports at top level create circular dependencies
   - Solution: Use `--no-type-checking` with lazy proxy imports

2. **NameError in get_type_hints()** ✅ FIXED:
   - Symptom: `NameError: name 'neuronxcc' is not defined`
   - Cause: TYPE_CHECKING imports not available at runtime
   - Solution: Use `--no-type-checking` - lazy proxies are placed in `globals()` where `get_type_hints()` can find them

3. **Union Type Assertion (NKI-specific)**:
   - Symptom: `AssertionError: Unexpected annotation! The first non-type hint is str | None`
   - Cause: NKI sema code doesn't handle `types.UnionType` returned by `get_type_hints()`
   - Solution: Avoid union type annotations on NKI-decorated functions, or fix neuronxcc's `sema.check_param_type`

4. **ModuleNotFoundError**:
   - Symptom: `ModuleNotFoundError: No module named 'np'`
   - Cause: Corrupted import aliases from previous RightTyper runs
   - Solution: Restore from backup and re-run

### Verification Commands

```bash
# Check a modified file has lazy proxy imports
head -60 neuronxcc/starfish/penguin/common.py
tail -50 neuronxcc/starfish/penguin/common.py  # Look for _LazyModule class

# Look for _LazyModule at end of file
grep -l "class _LazyModule" neuronxcc/**/*.py

# Count modified files
find neuronxcc -name "*.py.bak" | wc -l

# Verify get_type_hints works (should succeed after fix)
python3 -c "import typing; import neuronxcc.nki.compiler.backends.neuron.TraceKernel as TK; print(typing.get_type_hints(TK.TraceKernel.specialize_and_call))"
```

## Key Learnings

### 1. `typing.get_type_hints()` Behavior

**Critical insight**: `typing.get_type_hints()` evaluates annotation strings using `eval()` with the function's `__globals__` dict as the namespace. This means:

- Module-level `__getattr__` (PEP 562) does NOT help - it only triggers for attribute access on the module object itself (`module.attr`), not for dict lookups (`globals()['name']`)
- For `get_type_hints()` to resolve annotation strings like `neuronxcc.nki.Foo`, the name `neuronxcc` must exist directly in `globals()`
- The solution is to place proxy objects in `globals()` that trigger lazy imports when their attributes are accessed

### 2. Nested Module Import Behavior

When you `import neuronxcc.nki.compiler.backends.neuron.tensors`:
- Python creates `neuronxcc` module and adds `nki` as an attribute
- `neuronxcc.nki` gets `compiler` as attribute, etc.
- After the import, `neuronxcc.nki.compiler.backends.neuron.tensors` is fully accessible

This is why the `_LazyModule` proxy imports the deepest module path - it ensures all intermediate modules are populated as attributes on the top-level module.

### 3. Union Types and Runtime Type Checking

When `typing.get_type_hints()` resolves `str | None`:
- It returns a `types.UnionType` object, not a tuple of types
- Code that does `isinstance(hint, type)` will fail for union types
- This affects frameworks like NKI that validate type hints at runtime

**Pattern for handling union types**:
```python
import types
import typing

def get_individual_types(hint):
    if isinstance(hint, types.UnionType):
        return typing.get_args(hint)
    if typing.get_origin(hint) is typing.Union:
        return typing.get_args(hint)
    return (hint,)
```

### 4. Proxy Class Design for Lazy Imports

Key design decisions for `_LazyModule`:

1. **Use `__slots__`** - Prevents accidental attribute assignment
2. **Prefix internal attributes with `_rt_`** - Avoids conflicts with module attributes
3. **Use `object.__setattr__` and `object.__getattribute__`** - Bypasses `__getattr__` for internal state
4. **Import deepest path, return top-level module** - Ensures all intermediate modules are populated
5. **Cache the loaded module** - Import only happens once

### 5. Testing Strategy for Type Annotation Tools

When testing tools that add type annotations to large codebases:

1. **Create backups first** - Use `tar` or `rsync` for quick restore
2. **Test with tool running** - Verifies annotations are added correctly
3. **Test without tool** - Verifies annotated code works at runtime
4. **Check specific failure modes**:
   - `NameError` - Import not available
   - `ImportError` - Circular imports
   - `AssertionError` - Framework-specific type validation
5. **Isolate issues** - Distinguish between tool bugs and target codebase limitations
