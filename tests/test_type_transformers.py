import types
from righttyper.typeinfo import TypeInfo, AnyTypeInfo, NoneTypeInfo, UnknownTypeInfo, UnionTypeInfo
from righttyper.type_transformers import MakePickleableT, LoadTypeObjT, ExcludeTestTypesT
from righttyper.type_id import get_type_name


def test_load_type_obj():
    t = LoadTypeObjT().visit(TypeInfo('', 'foobar'))    # non-existant
    assert t.type_obj is None

    t = LoadTypeObjT().visit(TypeInfo('', 'int'))
    assert t.type_obj is int

    t = LoadTypeObjT().visit(TypeInfo('', 'None'))
    assert t.type_obj is types.NoneType

    t = LoadTypeObjT().visit(TypeInfo('not', 'modified', type_obj=int))
    assert str(t) == "not.modified"


class Foo: pass

def test_make_picklable_and_reload():
    for t_in in (int, types.FunctionType, Foo):
        t_out = LoadTypeObjT().visit(MakePickleableT().visit(get_type_name(t_in)))
        assert t_out.type_obj is t_in

    class ImLocal: pass

    t_out = LoadTypeObjT().visit(MakePickleableT().visit(get_type_name(ImLocal)))
    assert t_out.type_obj is None
    assert str(t_out) == f"{__name__}.test_make_picklable_and_reload.<locals>.ImLocal"


def test_exclude_test_types_exact_match():
    """Exact module name should be excluded."""
    tr = ExcludeTestTypesT({'_pytest'})
    t = tr.visit(TypeInfo('_pytest', 'SomeClass'))
    assert t == AnyTypeInfo


def test_exclude_test_types_submodule():
    """Submodule of test module should be excluded."""
    tr = ExcludeTestTypesT({'_pytest'})
    t = tr.visit(TypeInfo('_pytest.capture', 'EncodedFile'))
    assert t == AnyTypeInfo


def test_exclude_test_types_deep_submodule():
    """Deeply nested submodule should be excluded."""
    tr = ExcludeTestTypesT({'_pytest'})
    t = tr.visit(TypeInfo('_pytest.capture.foo.bar', 'SomeClass'))
    assert t == AnyTypeInfo


def test_exclude_test_types_no_match():
    """Non-test module should not be excluded."""
    tr = ExcludeTestTypesT({'_pytest', 'unittest'})
    t = tr.visit(TypeInfo('myapp.models', 'User'))
    assert t.module == 'myapp.models'
    assert t.name == 'User'


def test_exclude_test_types_partial_name_no_match():
    """Module starting with test module name but not a submodule should not match."""
    tr = ExcludeTestTypesT({'pytest'})
    t = tr.visit(TypeInfo('pytestfoo', 'Bar'))  # not pytest.foo
    assert t.module == 'pytestfoo'


# --- heuristic test module detection (detect_by_name=True) ---

def test_heuristic_test_prefix():
    """Module starting with test_ should be detected."""
    tr = ExcludeTestTypesT(set(), detect_by_name=True)
    assert tr.visit(TypeInfo('test_foo', 'Bar')) == AnyTypeInfo
    assert tr.visit(TypeInfo('test_foo.helpers', 'Baz')) == AnyTypeInfo

def test_heuristic_test_component():
    """Module with .test. or .tests. component should be detected."""
    tr = ExcludeTestTypesT(set(), detect_by_name=True)
    assert tr.visit(TypeInfo('myapp.test.fixtures', 'F')) == AnyTypeInfo
    assert tr.visit(TypeInfo('myapp.tests.helpers', 'H')) == AnyTypeInfo

def test_heuristic_test_suffix():
    """Module ending with _test should be detected."""
    tr = ExcludeTestTypesT(set(), detect_by_name=True)
    assert tr.visit(TypeInfo('myapp.foo_test', 'Bar')) == AnyTypeInfo

def test_heuristic_no_false_positive():
    """Regular modules should not be matched by heuristic."""
    tr = ExcludeTestTypesT(set(), detect_by_name=True)
    t = tr.visit(TypeInfo('myapp.models', 'User'))
    assert t.module == 'myapp.models'
    # 'contest' contains 'test' but isn't a test module
    t = tr.visit(TypeInfo('contest.main', 'Entry'))
    assert t.module == 'contest.main'
    # 'attest' ends with 'test' but isn't a component boundary
    t = tr.visit(TypeInfo('attest', 'Verifier'))
    assert t.module == 'attest'

def test_heuristic_disabled_by_default():
    """Without detect_by_name, heuristic patterns should not match."""
    tr = ExcludeTestTypesT(set())
    t = tr.visit(TypeInfo('test_foo', 'Bar'))
    assert t.module == 'test_foo'


def test_exclude_test_types_removes_from_union():
    """Test types in a union should be removed, not replaced with Any.

    Replacing with Any poisons the whole union (X | Any = Any), losing all
    useful type information.  Removing the test member preserves the
    non-test types."""
    tr = ExcludeTestTypesT({'tests'})
    real = TypeInfo('click', 'Context')
    fake = TypeInfo('tests.test_black', 'FakeContext')
    union = TypeInfo.from_set({real, fake})
    assert isinstance(union, UnionTypeInfo)

    result = tr.visit(union)
    # Should be just click.Context, not Any.
    assert result.module == 'click'
    assert result.name == 'Context'


def test_exclude_test_types_union_all_test():
    """If ALL members of a union are from test modules, result should be Unknown."""
    tr = ExcludeTestTypesT({'tests'})
    a = TypeInfo('tests.fixtures', 'FakeA')
    b = TypeInfo('tests.helpers', 'FakeB')
    union = TypeInfo.from_set({a, b})

    result = tr.visit(union)
    assert result == UnknownTypeInfo


def test_exclude_test_types_union_only_none_left():
    """If removing test types from a union leaves only None, result should be Unknown.

    Optional[TestFoo] should not narrow to None — we don't know the real
    non-None type, so dropping the annotation is safer than claiming it's
    always None.
    """
    tr = ExcludeTestTypesT({'tests'})
    fake = TypeInfo('tests.fixtures', 'FakeContext')
    union = TypeInfo.from_set({fake, NoneTypeInfo})

    result = tr.visit(union)
    assert result == UnknownTypeInfo


def test_exclude_test_types_union_preserves_multiple_non_test():
    """Non-test members of a union should all be preserved."""
    tr = ExcludeTestTypesT({'tests'})
    real1 = TypeInfo('click', 'Context')
    real2 = TypeInfo('click', 'Option')
    fake = TypeInfo('tests.test_black', 'FakeContext')
    union = TypeInfo.from_set({real1, real2, fake})

    result = tr.visit(union)
    assert isinstance(result, UnionTypeInfo)
    assert len(result.args) == 2
    names = {a.name for a in result.args if isinstance(a, TypeInfo)}
    assert names == {'Context', 'Option'}
