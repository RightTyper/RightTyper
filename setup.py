import setuptools
import sys
import os
import re
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


setuptools.setup(
    packages=['righttyper'],
    version=get_version(),
    long_description=long_description(),
    long_description_content_type="text/markdown",
)
