import types
from righttyper.typeinfo import TypeInfo, AnyTypeInfo
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


def test_exclude_test_types():
    """Test that ExcludeTestTypesT excludes test modules and submodules."""
    transformer = ExcludeTestTypesT({'pytest', 'myproject.test'})

    # Explicit test modules should be excluded
    t = transformer.visit(TypeInfo('pytest', 'SomeClass'))
    assert t == AnyTypeInfo

    # Submodules of explicit test modules should be excluded
    t = transformer.visit(TypeInfo('pytest.fixtures', 'FixtureClass'))
    assert t == AnyTypeInfo

    t = transformer.visit(TypeInfo('myproject.test.test_foo', 'MyClass'))
    assert t == AnyTypeInfo

    # Heuristic pattern detection should work
    # Modules containing .test. component
    t = transformer.visit(TypeInfo('other.test.module', 'SomeType'))
    assert t == AnyTypeInfo

    # Modules starting with test_
    t = transformer.visit(TypeInfo('test_something', 'SomeType'))
    assert t == AnyTypeInfo

    # Modules ending with _test
    t = transformer.visit(TypeInfo('something_test', 'SomeType'))
    assert t == AnyTypeInfo

    # Regular modules should NOT be excluded
    t = transformer.visit(TypeInfo('myproject.core', 'MyClass'))
    assert t.module == 'myproject.core'
    assert t.name == 'MyClass'

    # Modules that merely contain 'test' substring should NOT be excluded
    t = transformer.visit(TypeInfo('myproject.attestation', 'MyClass'))
    assert t.module == 'myproject.attestation'

    t = transformer.visit(TypeInfo('mytesting', 'MyClass'))
    assert t.module == 'mytesting'
