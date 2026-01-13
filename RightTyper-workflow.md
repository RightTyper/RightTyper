# RightTyper Type Annotation Workflow

## Overview

This document outlines a general workflow for using RightTyper to add type annotations to Python codebases and resolving subsequent Mypy errors.

## What is RightTyper?

RightTyper is a Python runtime type inference tool that:
- Observes actual types passed to/from functions during test execution
- Automatically generates type annotations with ~25% overhead
- Uses Python 3.12+ `sys.monitoring` for efficient observation
- Produces Mypy-compatible annotations
- Supports Python 3.12-3.14

## Architecture

```
CLI (righttyper.py)
    |
+---------------+------------------+
|  run command  |  process command |
+-------+-------+--------+---------+
        |                |
+-----------------------------------+
|   Type Observation System         |
|   recorder.py -> observations.py  |
|   type_id.py -> typeinfo.py       |
+---------------+-------------------+
                |
+-----------------------------------+
|   Code Transformation             |
|   unified_transformer.py (CST)    |
|   generalize.py, annotation.py    |
+---------------+-------------------+
                |
+-----------------------------------+
|   Output                          |
|   Python files, .pyi stubs, JSON  |
+-----------------------------------+
```

### Key Components

- **righttyper.py**: Main CLI with `run`, `process`, `coverage` commands using Click
- **righttyper_tool.py**: Sets up `sys.monitoring` hooks for function call/return observation
- **recorder.py**: Records function invocations and their argument/return types
- **observations.py**: Aggregates recorded type observations (probabilistic data structure)
- **type_id.py**: Converts runtime Python values to TypeInfo representations
- **unified_transformer.py**: libcst-based transformer that applies annotations to source
- **generalize.py**: Merges multiple observed types into unified type annotations
- **type_transformers.py**: Filters and simplifies types (mock resolution, test exclusion, depth limiting)

## RightTyper + Mypy Workflow

```
+------------------+     +------------------+     +------------------+
|  Run RightTyper  | --> |     Run Mypy     | --> |   Fix Errors     |
|    (annotate)    |     |    (validate)    |     |    (refine)      |
+------------------+     +------------------+     +------------------+
         |                       |                        |
         v                       v                        v
   Generates type          Detects type            Manual fixes or
   annotations from        inconsistencies         re-run with better
   runtime observation     and errors              test coverage
```

**Key insight:** RightTyper generates types from what it *observes*, not what's *correct*. Mypy then validates correctness.

## CLI Commands

RightTyper has three main commands:

1. **run** - Execute a script/module while collecting type information
2. **process** - Process previously collected observations (from `--only-collect`)
3. **coverage** - Compute annotation coverage statistics

## Recommended RightTyper Invocation

```bash
# Recommended options for annotating a codebase
python3 -m righttyper run \
    --overwrite \
    --no-use-typing-never \
    -m pytest

# Options explained:
# --overwrite             : Modify source files directly (creates .bak backups)
# --no-use-typing-never   : Avoid generating Never types (use Any instead)
# --no-exclude-test-files : Include test files in type inference (optional)
# -m pytest               : Run via pytest module
```

## Complete CLI Options Reference

### Basic Execution Options
- `-m, --module TEXT` - Run a Python module instead of a script (e.g., `-m pytest`)
- `--root DIRECTORY` - Process only files under a specific directory

### File Selection Options
- `--exclude-files GLOB` - Exclude files using fnmatch patterns (repeatable)
- `--exclude-test-files / --no-exclude-test-files` - Auto-exclude test modules (default: enabled)
- `--include-functions REGEX` - Only annotate functions matching regex (repeatable)

### Sampling & Threshold Options

Controls how aggressively RightTyper samples function calls:

- `--trace-min-samples INTEGER` - Minimum call traces before stopping (default: 5)
- `--trace-max-samples INTEGER` - Maximum call traces to sample (default: 25)
- `--trace-type-threshold FLOAT` - Threshold for stopping collection (default: 0.1)
- `--sampling / --no-sampling` - Enable/disable sampling (default: enabled)
- `--no-sampling-for REGEX` - Record EVERY invocation for matching functions

#### Container Sampling (for dict/list analysis)
- `--container-min-samples INTEGER` - Minimum container entries to sample (default: 15)
- `--container-max-samples INTEGER` - Maximum container entries to sample (default: 25)
- `--container-sample-limit [INTEGER|none]` - Max container elements (default: 1000)

### Type Complexity & Filtering Options

- `--type-depth-limit [INTEGER|none]` - Max depth of nested generics (default: unlimited)
- `--simplify-types / --no-simplify-types` - Simplify types like `int|bool|float -> float` (default: enabled)
- `--use-top-pct PCT` - Only use the PCT% most common call traces (default: 100)
- `--generalize-tuples N` - Generalize homogenous tuples to `tuple[T, ...]` if length >= N (default: 3)

### Type Version/Style Options
- `--python-version [3.9|3.10|3.11|3.12|3.13]` - Target Python version (default: 3.12)
- `--use-typing-never / --no-use-typing-never` - Emit `typing.Never` type (default: enabled for 3.11+)
- `--always-quote-annotations / --no-always-quote-annotations` - Quote all annotations (default: disabled)

### Test Type Exclusion & Mock Resolution
- `--exclude-test-types / --no-exclude-test-types` - Replace test types with `typing.Any` (default: enabled)
- `--resolve-mocks / --no-resolve-mocks` - Resolve mock objects to actual types (default: disabled)
- `--test-modules MODULE` - Additional modules to treat as test code

### Output Format Options
- `--output-files / --no-output-files` - Write annotated files (default: enabled)
- `--overwrite / --no-overwrite` - Overwrite original .py files (default: enabled)
- `--json-output` - Output annotations in JSON format to righttyper.json
- `--generate-stubs` - Generate .pyi stub files instead of modifying source
- `--ignore-annotations` - Ignore existing annotations and replace with inferred types
- `--only-update-annotations` - Update existing annotations but never add new ones

### Data Collection Options
- `--only-collect` - Save observations to `righttyper-N.rt` for later processing
- `--variables / --no-variables` - Observe and annotate module/local variables (default: enabled)

### Special Features
- `--infer-shapes` - Produce tensor shape annotations (requires jaxtyping package)
- `--infer-wrapped-return-type / --no-infer-wrapped-return-type` - Infer return type from wrapper functions (default: enabled)

### Coverage Command
```bash
python3 -m righttyper coverage --type [by-directory|by-file|summary] /path/to/project
```

## Output Files

| File | Purpose |
|------|---------|
| `righttyper.out` | Default diff-style output of type changes |
| `righttyper.json` | JSON format output (with `--json-output`) |
| `righttyper-N.rt` | Pickled observations (with `--only-collect`) |
| `righttyper-profiling.json` | Profiling data (with `--save-profiling`) |
| `*.pyi` | Stub files (with `--generate-stubs`) |
| `*.py.typed` | Modified files (with `--no-overwrite`) |
| `*.py.bak` | Backup of original files |

## TypeVar Inference Behavior

RightTyper infers TypeVar constraints when parameters have the same type across multiple calls:

```python
# Given these calls:
foo(1, 2)
foo(3.14, 2.71)

# RightTyper infers (note the TypeVar constraint):
def foo[T1: (float, int)](x: T1, y: T1) -> ...: ...

# NOT:
def foo(x: int | float, y: int | float) -> ...: ...
```

This TypeVar pattern captures the constraint that both params must have the same type in each call.

## Wrapped Function Type Propagation

RightTyper handles decorators where the wrapped function never executes (e.g., JIT compilers, `functools.wraps`):

1. **Detection**: `recorder.py` detects wrapped functions via:
   - `__call__` methods on objects with `__wrapped__` attribute
   - Regular functions with `__wrapped__` attribute
   - Wrapper functions created by `functools.wraps`

2. **Pending Traces Pattern**: Since wrapped functions don't execute, return types come from the wrapper's actual return value.

3. **Configurable via `--infer-wrapped-return-type`**:
   - Default (enabled): infer return type from wrapper's actual return value
   - Disabled: use `None` as placeholder return type

## Common Error Categories After RightTyper

### Category 1: Name Not Defined Errors
Types referenced in annotations that aren't imported at runtime. RightTyper may reference types that exist only during test execution.

**Fix:** Add missing imports or define type aliases. Consider `--always-quote-annotations`.

### Category 2: Never Type Issues
RightTyper infers `Never` when observing empty collections like `{}` or `[]`.

**Fix:** Use `--no-use-typing-never` flag.

### Category 3: Overly Complex Union Types
RightTyper observes all runtime variations and creates complex unions like:
```python
dict[str, list[dict[str, list[int]|str]]|list[dict[str, str]]|list[Any]|str]
```

**Fix:** Use `--type-depth-limit N` to limit nesting, or manually simplify.

### Category 4: Type Mismatches
Declared types don't match actual usage patterns.

**Fix:** Manual review - may indicate actual bugs or need for type refinement.

## Running the Tools

```bash
# Step 1: Clean previous artifacts
rm -f **/*.py.bak **/*.py.typed

# Step 2: Run RightTyper
python3 -m righttyper run --overwrite --no-use-typing-never -m pytest 2>&1 | tee righttyper.log

# Step 3: Run Mypy
python3 -m mypy src/ --ignore-missing-imports 2>&1 | tee mypy.errs

# Step 4: Analyze and fix errors
grep -c "error:" mypy.errs  # Count errors
```

## Workflow Examples

### Minimal usage (just see inferred types):
```bash
python3 -m righttyper run your_script.py
```

### Add types directly to source files:
```bash
python3 -m righttyper run --output-files --overwrite your_script.py
```

### Limit type complexity for readability:
```bash
python3 -m righttyper run --type-depth-limit=2 --output-files your_script.py
```

### Collect observations from multiple runs:
```bash
python3 -m righttyper run --only-collect test1.py
python3 -m righttyper run --only-collect test2.py
python3 -m righttyper process --output-files --overwrite
```

### Generate JAX/NumPy shape annotations:
```bash
python3 -m righttyper run --infer-shapes --output-files your_ml_script.py
```

---

## Problems Encountered / Future Work for RightTyper

### Problem 1: Overly Complex Inferred Types from JSON Parsing

**Issue:** When code parses JSON (e.g., `json.loads()`), RightTyper observes all possible runtime type variations and generates extremely complex union types:

```python
# RightTyper generates:
metadata: dict[str, list[dict[str, list[int]|str]]|list[dict[str, str]]|list[str]|list[Any]|str]

# What's practical:
metadata: dict[str, Any]
```

**Impact:** These complex types fail Mypy validation when used with methods like `str.join()` that expect simpler types.

**Potential Fix:** Use `--type-depth-limit N` to limit nesting depth, or add a `--max-union-complexity` option.

### Problem 2: Missing Runtime Imports in Annotations

**Issue:** RightTyper references type names that exist at runtime but aren't imported in the module being annotated. For example, if a function receives an object of type `Foo` from another module, RightTyper adds `param: Foo` but doesn't add the import.

**Impact:** Mypy reports "Name 'X' is not defined" errors.

**Workaround:** Use `--always-quote-annotations` to avoid some evaluation errors.

**Potential Fix:**
- Track where observed types originate and add necessary imports
- Or generate `TYPE_CHECKING` import blocks automatically

### Problem 3: Empty Collection Type Inference

**Issue:** Empty collections `[]`, `{}`, `()` are inferred as `list[Never]`, `dict[Never, Never]`, `tuple[()]`.

**Impact:** These types are technically correct but semantically useless - they describe collections that can never contain anything.

**Current Workaround:** Use `--no-use-typing-never` flag.

**Potential Enhancement:** Infer element types from how the collection is later used (e.g., if `.append(x)` is called, infer the list's element type from `x`).

### Problem 4: Complex Dictionary Return Types

**Issue:** Functions returning dictionaries with varied value types generate complex annotations:

```python
def get_config() -> dict[str, list[str] | str | int | None]:
```

**Impact:** Downstream code must handle all union variants or use type narrowing.

**Potential Fix:** Add `--simplify-dict-values` to collapse heterogeneous dict value types to `Any`.

### Problem 5: Wrapper/Decorator Function Attribute Access

**Issue:** When decorated functions have additional attributes added by wrappers (e.g., `func.grid`, `func.__wrapped__`), RightTyper annotates the function as `Callable[..., Any]` which doesn't have these attributes.

**Impact:** Mypy reports "Callable has no attribute 'X'" errors.

**Potential Fix:** Detect common decorator patterns and generate Protocol types or use `# type: ignore` comments.

### Problem 6: Circular Import Challenges

**Issue:** Running RightTyper on codebases with complex import structures can trigger circular import errors during the instrumented execution.

**Impact:** Tests fail to run, no type information collected.

**Potential Fix:** Better handling of import-time instrumentation to avoid triggering circular imports.

### Problem 7: Native Extension Module Types

**Issue:** Types from C extensions or native modules may not be fully introspectable, leading to incomplete or `Any` type annotations.

**Impact:** Loss of type safety for native module interactions.

**Potential Fix:** Allow users to provide stub hints for native modules.

---

## Recommendations for RightTyper Users

1. **Always use `--no-use-typing-never`** to avoid empty collection type issues

2. **Run comprehensive tests** - RightTyper only sees types from executed code paths

3. **Expect manual refinement** - RightTyper provides a starting point, not a final solution

4. **Use type depth limits** - Use `--type-depth-limit N` for complex nested types

5. **Review complex unions** - Simplify `A|B|C|D|E` to `A|B|Any` where practical

6. **Add missing imports** - Budget time for adding imports that RightTyper doesn't generate

7. **Use iteratively** - Run RightTyper -> Fix errors -> Improve tests -> Repeat

8. **Consider `--only-collect`** - For large codebases, collect observations from multiple test runs and combine them with `process`

9. **Use `--infer-shapes`** - For ML codebases with JAX/NumPy, enable tensor shape annotations
