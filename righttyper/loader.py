from typing import TYPE_CHECKING, Self
if TYPE_CHECKING:
    import _frozen_importlib
import ast
from importlib.abc import MetaPathFinder, Loader
from importlib import machinery
from importlib.resources.abc import TraversableResources
import functools
from pathlib import Path
import sys
import sysconfig
import typing

from righttyper.ast_instrument import GeneratorSendTransformer

class RightTyperLoader(Loader):
    def __init__(self, orig_loader: Loader, filename: Path):
        self.orig_loader = orig_loader  # original loader we're wrapping
        self.filename = filename

    # for compability with loaders supporting resources, used e.g. by sklearn
    def get_resource_reader(self, fullname: str) -> TraversableResources|None:
        if hasattr(self.orig_loader, 'get_resource_reader'):
            return self.orig_loader.get_resource_reader(fullname)
        return None

    def create_module(self, spec):
        return self.orig_loader.create_module(spec)

    def get_code(self, name=None):   # expected by pyrun
        tree = ast.parse(self.filename.read_bytes())
        tree = GeneratorSendTransformer().visit(tree)
        return compile(tree, str(self.filename), "exec")

    def exec_module(self, module):
        exec(self.get_code(), module.__dict__)


class RightTyperMetaPathFinder(MetaPathFinder):
    def __init__(self: Self) -> None:
        self._pylib_paths = (
            Path(sysconfig.get_path("stdlib")).resolve(),
            Path(sysconfig.get_path("purelib")).resolve(),
        )


    @functools.cache
    def _in_python_libs(self: "RightTyperMetaPathFinder", filename: Path) -> bool:
        if any(filename.is_relative_to(p) for p in self._pylib_paths):
            return True

        return False


    def find_spec(self: Self, fullname: str, path: None, target: None=None) -> "_frozen_importlib.ModuleSpec":
        for f in sys.meta_path:
            if isinstance(f, __class__) or not hasattr(f, "find_spec"):
                continue

            spec = f.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue

            # We need a python file to be able to instrument
            # TODO we could check for isinstance(spec.loader, machinery.SourceFileLoader)
            # but what about cached bytecode loaders?
            if (
                not spec.origin or spec.origin == 'built-in' or
                isinstance(spec.loader, machinery.ExtensionFileLoader)
            ):
                return None

            filename = Path(spec.origin)
            if filename.exists() and not self._in_python_libs(filename):
                spec.loader = RightTyperLoader(spec.loader, filename)

            return spec

        return None


class ImportManager:
    """A context manager that enables instrumentation while active."""

    def __init__(self: Self) -> None:
        self.mpf = RightTyperMetaPathFinder()


    def __enter__(self: Self) -> "ImportManager":
        sys.meta_path.insert(0, self.mpf)
        return self


    def __exit__(self: Self, *args: typing.Any) -> None:
        i = 0
        while i < len(sys.meta_path):
            if sys.meta_path[i] is self.mpf:
                sys.meta_path.pop(i)
                break
            i += 1
