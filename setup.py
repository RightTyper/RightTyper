import setuptools
import sys
import os
from pathlib import Path

try:
    import tomllib  # available in Python 3.11+
except ImportError:
    import tomli as tomllib


def get_version():
    return (
        tomllib.loads(Path("pyproject.toml").read_text())['tool']['righttyper_setup']['version'] + (
            '.dev' + Path('dev-build.txt').read_text().strip()
            if Path('dev-build.txt').exists()
            else ''
        )
    )


def cxx_version(v):
    return [f"-std={v}" if sys.platform != "win32" else f"/std:{v}"]


def platform_compile_args():
    # If flags are specified as a global env var, use them: this might happen
    # in a conda build, and is needed to override build configurations on osx
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


def bdist_wheel_options():
    options = {}

    # Build universal wheels on MacOS.
    if sys.platform == 'darwin' and \
       sum(arg == '-arch' for arg in platform_compile_args()) > 1:
        # On MacOS >= 11, all builds are compatible for a major MacOS version, so Python "floors"
        # all minor versions to 0, leading to tags like like "macosx_11_0_universal2". If you use
        # the actual (non-0) minor name in the build platform, pip doesn't install it.
        import platform
        v = platform.mac_ver()[0]
        major = int(v.split('.')[0])
        if major >= 11:
            v = f"{major}.0"
        options['plat_name'] = f"macosx-{v}-universal2"

    return options


setuptools.setup(
    version=get_version(),
    ext_modules= [
        setuptools.Extension(
            'righttyper.traverse',
            sources=['righttyper/traverse.cpp'],
            extra_compile_args=cxx_version('c++17') + platform_compile_args(),
            extra_link_args=platform_link_args(),
            py_limited_api=False,
            language='c++',
        )
    ],
    options={'bdist_wheel': bdist_wheel_options()}
)
