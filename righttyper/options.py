from dataclasses import dataclass
from typing import Sequence
import functools
import re

@dataclass
class Options:
    """Options for the run command."""

    script_dir: str = ""
    include_files_pattern: str = ""
    include_all: bool = False
    include_functions_pattern: tuple[str, ...] = tuple()
    target_overhead: float = 5.0
    infer_shapes: bool = False
    ignore_annotations: bool = False
    overwrite: bool = False
    output_files: bool = False
    generate_stubs: bool = False
    json_output: bool = False
    use_multiprocessing: bool = True
    sampling: bool = True
    no_sampling_for: str|None = None
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
    def test_modules_re(self) -> re.Pattern:
        """Returns a regular expression pattern to match test modules with."""
        return re.compile('|'.join([f"({m}(?:\\.|$))" for m in self.test_modules]))

    @functools.cached_property
    def no_sampling_for_re(self) -> re.Pattern|None:
        """Returns a regular expression pattern for no_sampling_for."""
        return re.compile(self.no_sampling_for) if self.no_sampling_for else None

options = Options()
