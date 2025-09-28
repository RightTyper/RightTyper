import setuptools
import sys
import os
import re
from pathlib import Path

try:
    import tomllib  # available in Python 3.11+
except ImportError:
    import tomli as tomllib

try:
    import pybind11
except Exception as exc:
    print("pybind11 is required to build this extension. pip install pybind11", file=sys.stderr)
    raise


def get_version():
    return (
        tomllib.loads(Path("pyproject.toml").read_text())['tool']['righttyper_setup']['version'] + (
            '.dev' + Path('dev-build.txt').read_text().strip()
            if Path('dev-build.txt').exists()
            else ''
        )
    )


def get_url():
    return tomllib.loads(Path("pyproject.toml").read_text())['project']['urls']['Repository']


def long_description():
    text = Path("README.md").read_text(encoding="utf-8")

    # Rewrite any relative paths to version-specific absolute paths,
    # so that they work from within PyPI
    sub = r'\1' + get_url() + "/blob/v" + get_version() + r'/\2'
    text = re.sub(r'(src=")((?!https?://))', sub, text)
    text = re.sub(r'(\[.*?\]\()((?!https?://))', sub, text)

    return text


def cxx_version(v):
    return [f"-std={v}" if sys.platform != "win32" else f"/std:{v}"]


def platform_compile_args():
    # If flags are specified as a global env var, use them: this happens in
    # the conda build, and is needed to override build configurations on osx
    if flags := os.environ.get("CXXFLAGS", "").split():
        return flags

    if sys.platform == 'darwin':
        # default to a multi-arch build
        return ['-arch', 'x86_64', '-arch', 'arm64', '-arch', 'arm64e']
    if sys.platform == 'win32':
        # avoids creating Visual Studio dependencies
        return ['/MT']
    return []


def platform_link_args():
    if sys.platform != 'win32':
        return platform_compile_args() # clang/gcc is used
    return []


setuptools.setup(
    packages=['righttyper'],
    version=get_version(),
    long_description=long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[
        setuptools.Extension(
            "righttyper.self_profiling",
            sources=["righttyper/self_profiling.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=cxx_version("c++17") + ["-g", "-O3"] + platform_compile_args(),
            extra_link_args=platform_link_args(),
            py_limited_api=False # doesn't work with pybind11
        )
    ]
)
