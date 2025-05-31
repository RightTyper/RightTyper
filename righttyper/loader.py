import ast
from importlib.abc import MetaPathFinder, Loader, ExecutionLoader, PathEntryFinder
from importlib import machinery
from importlib.resources.abc import TraversableResources
import importlib.util
import functools
from pathlib import Path
import sys
import sysconfig
import types
import typing

from righttyper.ast_instrument import instrument

class RightTyperLoader(ExecutionLoader):
    def __init__(self, fullname: str, path: Path, orig_loader: Loader|None=None, *, replace_dict: bool):
        self.fullname = fullname
        self.path = path.resolve()
        self.orig_loader = orig_loader
        self.replace_dict = replace_dict

    # for compability with loaders supporting resources, used e.g. by sklearn
    def get_resource_reader(self, fullname: str) -> TraversableResources|None:
        if self.orig_loader and hasattr(self.orig_loader, 'get_resource_reader'):
            return self.orig_loader.get_resource_reader(fullname)
        return None

    # for compatibility with older (deprecated) resource loading
    def get_data(self, path) -> str:
        if self.orig_loader and hasattr(self.orig_loader, 'get_data'):
            return self.orig_loader.get_data(path)
        raise IOError("get_data not supported")

    def get_filename(self, fullname: str) -> str:
        return str(self.path)

    def get_source(self, fullname: str) -> str:
        return str(self.path.read_bytes(), "utf-8")

    def get_code(self, fullname: str) -> types.CodeType:
        tree = ast.parse(self.get_source(fullname))
        tree = instrument(tree, replace_dict=self.replace_dict)
        return compile(tree, str(self.path), "exec")

    def create_module(self, spec):
        if self.orig_loader:
            return self.orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        exec(self.get_code(''), module.__dict__)


class RightTyperMetaPathFinder(MetaPathFinder):
    def __init__(self: typing.Self, *, replace_dict: bool) -> None:
        self._pylib_paths = {
            Path(sysconfig.get_path(p)).resolve()
            for p in ('stdlib', 'platstdlib', 'purelib', 'platlib')
        }
        self.replace_dict = replace_dict


    @functools.cache
    def _in_python_libs(self: typing.Self, filename: Path) -> bool:
        if any(filename.is_relative_to(p) for p in self._pylib_paths):
            return True

        return False


    def find_spec(
        self: typing.Self,
        fullname: str,
        path: typing.Sequence[str]|None,
        target: types.ModuleType|None = None
    ) -> machinery.ModuleSpec | None:
        # Enlist the help of other loaders to get a spec
        for f in sys.meta_path:
            if (
                isinstance(f, self.__class__) or
                not hasattr(f, "find_spec") or
                (spec := f.find_spec(fullname, path, target)) is None or
                spec.loader is None
            ):
                # this loader can't help us, try another
                continue

            if (
                spec.origin and
                spec.origin != 'built-in' and
                not isinstance(spec.loader, machinery.ExtensionFileLoader) and
                (filename := Path(spec.origin)).exists() and
                filename.suffix == '.py' and
                not self._in_python_libs(filename)
            ):
                # For those that look like we can load, insert our loader
                spec.loader = RightTyperLoader(fullname, filename, spec.loader, replace_dict=self.replace_dict)

            return spec

        return None


class RightTyperPathEntryFinder(PathEntryFinder):
    def __init__(self, filename, *, replace_dict: bool):
        self.filename = filename
        self.replace_dict = replace_dict

    def find_spec(self, fullname: str, target: types.ModuleType|None=None) -> machinery.ModuleSpec|None:
        # we want to intercept runpy's loading of some script (".py") as __main__
        if fullname == "__main__":
            return importlib.util.spec_from_loader(
                fullname,
                RightTyperLoader(fullname, Path(self.filename), replace_dict=self.replace_dict)
            )

        return None


class PathEntryHook:
    def __init__(self, *, replace_dict: bool):
        self.replace_dict = replace_dict

    def __call__(self, filename: str) -> PathEntryFinder:
        # we want to intercept runpy's loading of some script (".py") as __main__
        if filename.endswith(".py"):
            return RightTyperPathEntryFinder(filename, replace_dict=self.replace_dict)

        raise ImportError()


def pytest_patcher():
    """Patches pytest to allow instrumentation."""

    orig_rewrite_asserts: typing.Any = None
    orig_read_pyc: typing.Any = None
    orig_write_pyc: typing.Any = None

    try:
        import _pytest.assertion.rewrite as pyrewrite

        orig_rewrite_asserts = pyrewrite.rewrite_asserts
        orig_read_pyc = pyrewrite._read_pyc
        orig_write_pyc = pyrewrite._write_pyc
    except ModuleNotFoundError:
        pass

    def rewrite_asserts(mod: ast.Module, *args, **kwargs):
        # wrap assertion rewriter to do our own instrumentation on the AST
        assert isinstance(mod, ast.Module)
        return orig_rewrite_asserts(instrument(mod), *args, **kwargs)

    def read_pyc(*args, **kwargs):
        # don't read any cached pyc, forcing it to go to source
        return None 

    def write_pyc(*args, **kwargs):
        # don't save our patched bytecode... this might be slow
        pass

    if orig_rewrite_asserts:
        pyrewrite.rewrite_asserts = rewrite_asserts
        pyrewrite._read_pyc = read_pyc
        pyrewrite._write_pyc = write_pyc

    yield   # return control a la pytest fixture

    if orig_rewrite_asserts:
        # now clean up
        pyrewrite.rewrite_asserts = orig_rewrite_asserts
        pyrewrite._read_pyc = orig_read_pyc
        pyrewrite._write_pyc = orig_write_pyc


class ImportManager:
    """A context manager that enables instrumentation while active."""

    def __init__(self, *, replace_dict: bool) -> None:
        self.mpf = RightTyperMetaPathFinder(replace_dict=replace_dict)
        self.peh = PathEntryHook(replace_dict=replace_dict)


    def __enter__(self) -> typing.Self:
        sys.meta_path.insert(0, self.mpf)
        sys.path_hooks.insert(0, self.peh)
        self.pytest_patcher = pytest_patcher()
        next(self.pytest_patcher)
        return self


    def __exit__(self, *args) -> None:
        next(self.pytest_patcher, None)
        sys.path_hooks.remove(self.peh)
        sys.meta_path.remove(self.mpf)
