from dataclasses import dataclass
from typing import Sequence, Any
import functools
import re
import click


@dataclass
class OutputOptions:
    """Output options; see the click options in righttyper.py"""

    overwrite: bool = True
    output_files: bool = True
    ignore_annotations: bool = False
    only_update_annotations: bool = False
    generate_stubs: bool = False
    json_output: bool = False
    use_multiprocessing: bool = True
    type_depth_limit: int|None = None
    use_typing_union: bool = False
    use_typing_self: bool = False
    use_typing_never: bool = False
    inline_generics: bool = False
    use_top_pct: int = 100
    simplify_types: bool = True
    exclude_test_types: bool = True
    always_quote_annotations: bool = False


    def process_args(self, kwargs: dict[str, Any]) -> None:
        if kwargs['ignore_annotations'] and kwargs['only_update_annotations']:
            raise click.UsageError("Options --ignore-annotations and --only-update-annotations are mutually exclusive.")

        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)

        python_version = kwargs['python_version']
        use_typing_never = kwargs['use_typing_never']

        self.use_typing_union = python_version < (3, 10)
        self.use_typing_self = python_version >= (3, 11)
        self.use_typing_never = python_version >= (3, 11) and use_typing_never
        self.inline_generics = python_version >= (3, 12)


def _merge_regexes(patterns: Sequence[str]) -> re.Pattern[str]|None:
    """Merges multiple regular expressions, returning a compiled pattern or,
       if the tuple is empty, returns None.
    """
    return re.compile('|'.join([f"(?:{p})" for p in patterns])) if patterns else None


@dataclass
class RunOptions:
    """Options for the run command; see the click options in righttyper.py"""

    script_dir: str = ""
    exclude_files: tuple[str, ...] = ()
    exclude_test_files: bool = True
    include_functions: tuple[str, ...] = ()
    poisson_sample_rate: float = 2.0  # Expected capture windows per second
    poisson_warmup_samples: int = 5   # Capture first N samples immediately before Poisson timing
    infer_shapes: bool = False
    sampling: bool = True
    no_sampling_for: tuple[str, ...] = ()
    replace_dict: bool = False
    container_min_samples: int = 15
    container_max_samples: int = 50
    container_type_threshold: float = .1
    container_sample_limit: int|None = None
    container_window_size: int = 20  # Sliding window size for Good-Turing decision
    resolve_mocks: bool = False
    test_modules: tuple[str, ...] = ('pytest', '_pytest', 'py.test', 'unittest')
    adjust_type_names: bool = True
    variables: bool = True
    save_profiling: str|None = None
    allow_runtime_exceptions: bool = False
    generalize_tuples: int = 3


    def process_args(self, kwargs: dict[str, Any]) -> None:
        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)


    @functools.cached_property
    def include_functions_re(self) -> re.Pattern[str]|None:
        """Returns a regular expression pattern for no_sampling_for."""
        return _merge_regexes(self.include_functions)

    @functools.cached_property
    def no_sampling_for_re(self) -> re.Pattern[str]|None:
        """Returns a regular expression pattern for no_sampling_for."""
        return _merge_regexes(self.no_sampling_for)

run_options = RunOptions()
output_options = OutputOptions()
