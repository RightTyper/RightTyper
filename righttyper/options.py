from dataclasses import dataclass

@dataclass
class RunOptions:
    script_dir: str = ""
    include_files_pattern: str = ""
    include_all: bool = False
    include_functions_pattern: tuple[str, ...] = tuple()
    target_overhead: float = 5.0
    infer_shapes: bool = False
    ignore_annotations: bool = False
    srcdir: str = ""
    sampling: bool = True
    replace_dict: bool = False
    container_sample_limit: int = 1000
    use_typing_union: bool = False
    use_typing_self: bool = False
    use_typing_never: bool = False
    inline_generics: bool = False
    use_top_pct: int = 80

run_options = RunOptions()
