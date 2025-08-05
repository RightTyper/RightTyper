from dataclasses import dataclass

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
    use_multiprocessing: bool = True
    sampling: bool = True
    replace_dict: bool = False
    container_sample_limit: int = 1000
    type_depth_limit: int|None = None
    use_typing_union: bool = False
    use_typing_self: bool = False
    use_typing_never: bool = False
    inline_generics: bool = False
    only_update_annotations: bool = False
    use_top_pct: int = 80

options = Options()
