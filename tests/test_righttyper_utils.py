import functools

from unittest import mock

from righttyper.righttyper_utils import is_test_module, unwrap


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


def test_unwrap_terminates_on_synthesized_wrapped():
    """mock's _Call answers __wrapped__ with a brand-new child every time, so the
    id-based cycle guard never fires; without the depth cap this allocates until
    the process is OOM-killed.  The end-to-end shape is test_issue_193_mock_in_class_dict;
    this pins it in microseconds.  See #193.
    """
    assert unwrap(mock.call.patched) is None


def test_unwrap_follows_a_wraps_chain():
    def inner():
        pass

    @functools.wraps(inner)
    def outer():
        pass

    assert unwrap(outer) is inner


def test_unwrap_follows_a_decorated_bound_method():
    """unwrap() is handed bound methods (type_id.find_function, the CALL handler),
    not just class __dict__ entries.
    """
    def deco(fn):
        @functools.wraps(fn)
        def w(*args, **kwargs):
            return fn(*args, **kwargs)
        return w

    class C:
        @deco
        def m(self):
            pass

    assert unwrap(C().m) is C.__dict__["m"].__wrapped__


def test_unwrap_returns_none_on_a_cycle():
    def a():
        pass

    def b():
        pass

    a.__wrapped__ = b   # type: ignore[attr-defined]
    b.__wrapped__ = a   # type: ignore[attr-defined]

    assert unwrap(a) is None
