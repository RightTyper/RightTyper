import ast
import importlib.util
import os
import pathlib
import platform
import runpy
import subprocess
import sys
import sysconfig
from functools import lru_cache
from importlib.abc import Loader, MetaPathFinder


@lru_cache()
def get_homebrew_cellar_path() -> str:
    """Returns the Homebrew Cellar path by running `brew --cellar`."""
    try:
        result = subprocess.run(["brew", "--cellar"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""

    
@lru_cache()
def is_system_installed_package_file(file_path: str) -> bool:
    """
    Checks whether a given file belongs to a system-installed package, including
    Python's standard library, installed packages in site-packages, and platform-specific
    directories such as lib-dynload and Homebrew paths.
    
    Args:
        file_path (str): The absolute path to the file to check.
    
    Returns:
        bool: True if the file belongs to a system-installed package, False otherwise.
    """
    # Normalize the input file path
    file_path = os.path.normcase(os.path.abspath(file_path))

    # Get standard directories where system packages are installed
    package_dirs = [
        sysconfig.get_path('purelib'),   # Site-packages
        sysconfig.get_path('platlib'),   # Platform-specific site-packages
        sysconfig.get_path('stdlib'),    # Standard library
        sysconfig.get_path('platstdlib') # Platform-specific standard library
    ]

    # Get the dynamic library directory for platform-specific modules (like _contextvars.so)
    dynload_dir = os.path.join(sysconfig.get_path('platstdlib'), 'lib-dynload')
    package_dirs.append(dynload_dir)

    # Also add sys.prefix as a fallback for virtual environments or custom installs
    package_dirs.append(sys.prefix)

    # Detect if we are running on macOS
    if platform.system() == "Darwin":
        # Check if Python is installed via Homebrew by checking if 'brew' is available and valid
        brew_cellar_path = get_homebrew_cellar_path()
        if brew_cellar_path:
            package_dirs.append(brew_cellar_path)

    # Normalize and clean the paths for comparison
    package_dirs = [os.path.normcase(os.path.abspath(path)) for path in package_dirs if path]

    # Check if the file resides within any of the identified system directories
    for package_dir in package_dirs:
        if file_path.startswith(package_dir):
            return True

    return False


class DictTransformer(ast.NodeTransformer):
    def visit_Module(self, node):
        """Insert import at the top of the module only if necessary."""
        has_random_dict_import = any(
            isinstance(n, ast.ImportFrom) and n.module == 'righttyper.random_dict' for n in node.body
        )

        if not has_random_dict_import:
            import_node = ast.ImportFrom(
                module='righttyper.random_dict',
                names=[ast.alias(name='RandomDict', asname=None)],
                level=0
            )
            # Insert the import statement at the beginning of the module
            node.body.insert(0, import_node)

        # Continue with other transformations
        self.generic_visit(node)
        return node

    def visit_Dict(self, node):
        # Transform dictionary literals to RandomDict()
        self.generic_visit(node)
        if not node.keys and not node.values:
            # Empty dict literal {}
            return ast.Call(func=ast.Name(id='RandomDict', ctx=ast.Load()), args=[], keywords=[])
        else:
            # Non-empty dict literal {k1: v1, k2: v2, ...}
            dict_call = ast.Dict(keys=node.keys, values=node.values)
            return ast.Call(func=ast.Name(id='RandomDict', ctx=ast.Load()), args=[dict_call], keywords=[])

    def visit_DictComp(self, node):
        # Transform dictionary comprehensions to RandomDict()
        self.generic_visit(node)
        return ast.Call(func=ast.Name(id='RandomDict', ctx=ast.Load()), args=[node], keywords=[])

    def visit_Call(self, node):
        # Replace calls to dict() with RandomDict()
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'dict':
            node.func.id = 'RandomDict'
        return node

class TransformingLoader(Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        # Apply transformation only if the module is in the user code path
        if module.__file__ and module.__file__.endswith(os.extsep + "py"):
            if not is_system_installed_package_file(module.__file__):
                source = self.loader.get_source(module.__name__)
                if source is None:
                    return
                tree = ast.parse(source)
                transformer = DictTransformer()
                tree = transformer.visit(tree)
                ast.fix_missing_locations(tree)
                code = compile(tree, filename=module.__file__, mode='exec')
                exec(code, module.__dict__)
                return
        # If not in user code path, run the module without transformation
        self.loader.exec_module(module)

class TransformingFinder(MetaPathFinder):
    def __init__(self):
        self._processed_modules = set()

    def find_spec(self, fullname, path, target=None):
        if fullname in self._processed_modules:
            return None
        self._processed_modules.add(fullname)
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec and spec.loader:
                spec.loader = TransformingLoader(spec.loader)
                return spec
        return None

def transform_and_run_script(script_path):
    """Transform and run a Python script using AST transformation."""
    with open(script_path, 'r') as file:
        source = file.read()

    # Only apply transformation if the script is within the user code path
    if not is_system_installed_package_file(os.path.abspath(script_path)):
        tree = ast.parse(source)
        transformer = DictTransformer()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        code = compile(tree, filename=script_path, mode='exec')
        namespace = {}
        exec(code, namespace)
    else:
        # Fallback to default runpy if not in user path
        runpy.run_path(script_path, run_name="__main__")

def replace_dicts():
    """Replace dict with RandomDict and apply transformation to modules and scripts."""
    sys.meta_path.insert(0, TransformingFinder())
    # Monkey-patch runpy.run_path to apply AST transformation to scripts
    original_run_path = runpy.run_path

    def custom_run_path(script_path, init_globals=None, run_name=None):
        if not is_system_installed_package_file(os.path.abspath(script_path)):
            transform_and_run_script(script_path)
        else:
            return original_run_path(script_path, init_globals=init_globals, run_name=run_name)

    runpy.run_path = custom_run_path
