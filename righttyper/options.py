from dataclasses import dataclass
from typing import Sequence
import functools
import re


def _merge_regexes(patterns: tuple[str, ...]) -> re.Pattern|None:
    """Merges multiple regular expressions, returning a compiled pattern or,
       if the tuple is empty, returns None.
    """
    return re.compile('|'.join([f"{(p)}" for p in patterns])) if patterns else None


@dataclass
class Options:
    """Options for the run command."""

    script_dir: str = ""
    include_files: tuple[str, ...] = ()
    include_all: bool = False
    include_functions: tuple[str, ...] = ()
    target_overhead: float = 5.0
    infer_shapes: bool = False
    ignore_annotations: bool = False
    overwrite: bool = False
    output_files: bool = False
    generate_stubs: bool = False
    json_output: bool = False
    use_multiprocessing: bool = True
    sampling: bool = True
    no_sampling_for: tuple[str, ...] = ()
    replace_dict: bool = False
    container_sample_limit: int = 1000
    type_depth_limit: int|None = None
    use_typing_union: bool = False
    use_typing_self: bool = False
    use_typing_never: bool = False
    inline_generics: bool = False
    only_update_annotations: bool = False
    use_top_pct: int = 80
    exclude_test_types: bool = True
    resolve_mocks: bool = False
    test_modules: tuple[str, ...] = ('pytest', '_pytest', 'py.test', 'unittest')


    @functools.cached_property
    def include_files_re(self) -> re.Pattern|None:
        """Returns a regular expression pattern for no_sampling_for."""
        return _merge_regexes(self.include_files)

    @functools.cached_property
    def include_functions_re(self) -> re.Pattern|None:
        """Returns a regular expression pattern for no_sampling_for."""
        return _merge_regexes(self.include_functions)

    @functools.cached_property
    def test_modules_re(self) -> re.Pattern|None:
        """Returns a regular expression pattern to match test modules with."""
        # Escape dots and enforce module path boundaries
        return _merge_regexes([f"{m.replace('.', r'\.')}(?:\\.|$)" for m in self.test_modules])

    @functools.cached_property
    def no_sampling_for_re(self) -> re.Pattern|None:
        """Returns a regular expression pattern for no_sampling_for."""
        return _merge_regexes(self.no_sampling_for)

options = Options()
