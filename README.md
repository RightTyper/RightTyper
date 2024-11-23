# RightTyper

RightTyper is a Python tool that generates types for your function
arguments and return values. RightTyper lets your code run at nearly full speed with
almost no memory overhead. As a result, you won't experience slow
downs in your code or large memory consumption while using it,
allowing you to integrate it with your standard tests and development
process. By virtue of its design, and in a significant departure from previous approaches,
RightTyper only captures the most commonly used types,
letting a type checker like `mypy` detect possibly incorrect type mismatches in your code.

You can run RightTyper with arbitrary Python programs and it will generate
types for every function that gets executed. It works great in combination with [pytest](https://docs.pytest.org/):

```bash
python3 -m righttyper -m pytest --continue-on-collection-errors /your/test/dir
```

In addition to generating types, RightTyper has the following features:

* It efficiently computes type annotation "coverage" for a file or directory of files
* It infers shape annotations for NumPy/JAX/PyTorch tensors, compatible with [`jaxtyping`](https://docs.kidger.site/jaxtyping/) and [`beartype`](https://github.com/beartype/beartype) or [`typeguard`](https://typeguard.readthedocs.io/en/latest/).


## Usage

To use RightTyper, simply run your script with `righttyper` instead of `python3`:

```bash
righttyper your_script.py [args...]
```

This will execute `your_script.py` with RightTyper's monitoring
enabled. The type signatures of all functions will be recorded and
output to a file named `righttyper.out`. The file contains, for every
function, the signature, and a diff of the original function with the
annotated version. It also optionally (with the `--infer-shapes` flag)
generates `jaxtyping`-compatible shape
annotations for NumPy/JAX/PyTorch tensors. Below is an example:

```diff
test-hints.py:
==============

def barnacle(x: numpy.ndarray) -> numpy.ndarray: ...

- def barnacle(x):
+ def barnacle(x: numpy.ndarray) -> numpy.ndarray:

# Shape annotations
@beartype
def barnacle(x: Float[numpy.ndarray, "10 dim0"]) -> Float[numpy.ndarray, "dim0"]: ...

def fooq(x: int, y: str) -> bool: ...

- def fooq(x: int, y) -> bool:
+ def fooq(x: int, y: str) -> bool:
?                   +++++
```

Below is the full list of options:

```
Usage: python -m righttyper [OPTIONS] [SCRIPT] [ARGS]...

  RightTyper efficiently generates types for your function arguments and
  return values.

Options:
  --all-files                     Process any files encountered, including in
                                  libraries (except for those specified in
                                  --include-files)
  --include-files TEXT            Include only files matching the given regex
                                  pattern.
  --infer-shapes                  Produce tensor shape annotations (compatible
                                  with jaxtyping).
  --srcdir DIRECTORY              Use this directory as the base for imports.
  --overwrite / --no-overwrite    Overwrite files with type information.
                                  [default: no-overwrite]
  --output-files / --no-output-files
                                  Output annotated files (possibly
                                  overwriting, if specified).  [default: no-
                                  output-files]
  --ignore-annotations            Ignore existing annotations and overwrite
                                  with type information.
  -m, --module                    Run the script as a module.
  --verbose                       Print diagnostic information.
  --generate-stubs                Generate stub files (.pyi).
  --type-coverage-by-directory DIRECTORY
                                  Report per-directory type annotation
                                  coverage for all Python files in a directory
                                  and its children.
  --type-coverage-by-file DIRECTORY
                                  Report per-file type annotation coverage for
                                  all Python files in a directory or its
                                  children.
  --type-coverage-summary DIRECTORY
                                  Report uncovered and partially covered files
                                  and functions when performing type
                                  annotation coverage analysis.
  --version                       Show the version and exit.
  --target-overhead FLOAT         Target overhead, as a percentage (e.g., 5).
  --help                          Show this message and exit.
```

## `righttyper`: high performance

In the below example drawn from the pyperformance benchmark suite,
`monkeytype` runs 40x slower than the original program or when
running with `righttyper` (which runs under 3% slower).

```bash
% python3 bm_mdp          
Time elapsed:  6.106977417017333
% righttyper bm_mdp
Time elapsed:  6.299191833997611
% monkeytype run bm_mdp
Time elapsed:  184.57902495900635
# actual time elapsed was 275 seconds, spent post-processing
```

# `righttyper`: low memory consumption

With `monkeytype`, this program also consumes 5GB of RAM; the original
consumes just 21MB. That's an over **200x** increase in memory
consumption. `monkeytype` also leaves behind a 3GB SQLite file.

By contrast, `righttyper`'s memory consumption is just a small
increment over the original program: it consumes about 24MB, just 15%
more.

_NOTE: this is an alpha release and should not be considered production ready._

## Requirements

- Python 3.12 or higher

## How it works

Monkeytype is slow for several reasons. First, it uses Python's `setprofile` functionality
to track every single function call and return, gathers types for all
arguments and the return value. It checks every element of every argument (an O(N) operation).
Finally, it writes these into a SQLite database for later post-processing.

By contrast, RightTyper dynamically applies and removes type checking. It also only randomly samples arguments, letting
it track argument types with low overhead. RightTyper always checks the very first invocation and exit of every function.
It then performs _adaptive sampling_: RightTyper tracks its overhead of operation.
As long as instrumentation overhead remains below a target (configurable with `--target-overhead`, by default 5\%),
it keeps tracking functions. When necessary to reduce overhead, it reduces its rate of function tracking. Whenever
average overhead remains below the target, it continues function tracking.
