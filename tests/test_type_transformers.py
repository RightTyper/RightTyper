import types
from righttyper.typeinfo import TypeInfo
from righttyper.type_transformers import MakePickleableT, LoadTypeObjT
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
