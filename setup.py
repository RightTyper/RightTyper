import setuptools
import sys
import os

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


setuptools.setup(
    ext_modules= [
        setuptools.Extension(
            'righttyper.traverse',
            sources=['righttyper/traverse.cpp'],
            extra_compile_args=cxx_version('c++17') + platform_compile_args(),
            language='c++',
            py_limited_api=True
        )
    ]
)
