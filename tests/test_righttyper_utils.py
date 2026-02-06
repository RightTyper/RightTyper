from righttyper.righttyper_utils import is_test_module


def test_is_test_module_exact():
    """Exact module names from default test_modules should match."""
    assert is_test_module('_pytest') == True
    assert is_test_module('pytest') == True
    assert is_test_module('unittest') == True


def test_is_test_module_submodule():
    """Submodules of test modules should match."""
    assert is_test_module('_pytest.capture') == True
    assert is_test_module('pytest.fixtures') == True


def test_is_test_module_not_test():
    """Non-test modules should not match."""
    assert is_test_module('myapp') == False
    assert is_test_module('myapp.tests') == False  # 'tests' != 'pytest'


def test_is_test_module_partial_name():
    """Module starting with test module name but not a submodule should not match."""
    assert is_test_module('pytestfoo') == False  # not a submodule
