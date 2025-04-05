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


setuptools.setup(
    packages=['righttyper'],
    version=get_version(),
)
