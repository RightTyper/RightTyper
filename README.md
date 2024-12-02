# RightTyper

![Downloads](https://static.pepy.tech/badge/righttyper)[![Downloads](https://static.pepy.tech/badge/righttyper/month)](https://pepy.tech/project/righttyper) 

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


## Performance Comparison

The graph below presents the overhead of using RightTyper versus two
previous tools, MonkeyType and PyAnnotate, across a range of
benchmarks. On average, RightTyper imposes only 30% overhead compared
to running plain Python ("none"). On one popular package (black),
RightTyper imposes only 20% overhead, while RightTyper slows down
execution by over 37x. In extreme cases, MonkeyType runs over 3,000x
slower than RightTyper.

![Overhead](https://github.com/RightTyper/RightTyper/blob/main/docs/benchmark_comparison_execution_times.png)

## Usage

Install RightTyper from `pip` as usual:

```bash
python3 -m pip install righttyper
```

To use RightTyper, simply run your script with `python3 -m righttyper` instead of `python3`:

```bash
python3 -m righttyper your_script.py [args...]
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

To add type hints directly to your code, use this command:

```bash
python3 -m righttyper --output-files --overwrite your_script.py [args...]
```

To do the same with `pytest`:

```bash
python3 -m righttyper --output-files --overwrite -m pytest your_directory
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

