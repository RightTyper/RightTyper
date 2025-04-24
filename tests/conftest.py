import pytest

@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_makereport(item, call):
    """Saves the test result for use within a fixture teardown."""
    report = yield
    item._report = report
    return report

def pytest_addoption(parser):
    parser.addoption(
        "--no-mypy",
        action="store_true",
        default=False,
        help="Disable running mypy on test outputs"
    )
