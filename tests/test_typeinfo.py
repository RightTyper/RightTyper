from righttyper.righttyper_types import TypeInfo, TypeInfoSet
import righttyper.typeinfo


def ti(name: str, **kwargs) -> TypeInfo:
    return TypeInfo(module='', name=name, **kwargs)


def generalize(samples):
    result = righttyper.typeinfo.generalize(samples)
    if result:
        class Renamer(TypeInfo.Transformer):
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if node.typevar_index:
                    return TypeInfo(
                        module='',
                        name=f"T{node.typevar_index}",
                        typevar_index=node.typevar_index
                    )
                return super().visit(node)

        return [str(Renamer().visit(r)) for r in result]

    return result


def test_empty():
    assert generalize([]) == []
    assert generalize([tuple()]) == []


def test_single_sample():
    assert generalize([(ti('int'), ti('float'), ti('str'))]) == ['int', 'float', 'str']


def test_varied_length_samples():
    samples = [
        (ti('int'), ti('int')),
        (ti('int'), ti('int'), ti('int')),
    ]
    # TODO raise exception instead?
    assert generalize(samples) is None


def test_uniform_single_type():
    samples = [
        (ti('int'), ti('int'), ti('int')),
        (ti('bool'), ti('bool'), ti('bool')),
        (ti('float'), ti('float'), ti('float'))
    ]
    assert generalize(samples) == ['T1', 'T1', 'T1']


def test_uniform_single_type_with_generic():
    samples = [
        (ti('int'), ti('int')),
        (ti('bool'), ti('bool')),
        (ti('X', args=(ti('foo'),)), ti('X', args=(ti('foo'),))),
    ]

    assert generalize(samples) == ['T1', 'T1']


def test_first_same_then_different():
    samples = [
        (ti('int'), ti('int')),
        (ti('bool'), ti('bool')),
        (ti('int'), ti('bool'))
    ]
    assert generalize(samples) == ['bool|int', 'bool|int']


def test_mixed_with_constant_types():
    samples = [
        (ti('int'), ti('str'), ti('int')),
        (ti('bool'), ti('str'), ti('float')),
        (ti('float'), ti('str'), ti('bool'))
    ]
    assert generalize(samples) == ['bool|float|int', 'str', 'bool|float|int']


def test_shared_variability():
    samples = [
        (ti('int'), ti('int'), ti('bool'), ti('int')),
        (ti('float'), ti('float'), ti('bool'), ti('float'))
    ]
    assert generalize(samples) == ['T1', 'T1', 'bool', 'T1']


def test_all_distinct_types():
    samples = [
        (ti('int'), ti('str'), ti('float'), ti('bool')),
        (ti('float'), ti('str'), ti('bool'), ti('int'))
    ]
    assert generalize(samples) == ['float|int', 'str', 'bool|float', 'bool|int']


def test_generic():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
    ]
    assert generalize(samples) == ['T1', 'list[T1]']


def test_generic_not_generalizable():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
        (ti('bool'), TypeInfo.from_type(type(None))),
    ]
    assert generalize(samples) == ['bool|float|int', 'list[float|int]|None']

    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
        (ti('bool'), ti('list', args=(ti('int'),)))
    ]
    assert generalize(samples) == ['bool|float|int', 'list[float|int]']

    samples = [
        (ti('int'), ti('tuple', args=(ti('int'),))),
        (ti('float'), ti('tuple', args=(ti('float'),))),
        (ti('bool'), ti('tuple', args=(ti('bool'), ti('int'))))
    ]
    assert generalize(samples) == ['bool|float|int', 'tuple[bool, int]|tuple[float|int]']


def test_generic_among_options():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('X', args=(ti('foo'),)), ti('list', args=(ti('X', args=(ti('foo'),)),))),
    ]
    assert generalize(samples) == ['T1', 'list[T1]']


def test_generic_within_args():
    samples = [
        (ti('tuple', args=(ti('int'),)), ti('list', args=(ti('int'),))),
        (ti('tuple', args=(ti('float'),)), ti('list', args=(ti('float'),))),
    ]
    assert generalize(samples) == ['tuple[T1]', 'list[T1]']


def test_generic_with_string():
    from typing import Any

    samples: Any = [
        (ti('int'), ti('X', args=(ti('int'), '"foo"'))),
        (ti('bool'), ti('X', args=(ti('bool'), '"bar"'))),
    ]
    assert generalize(samples) == ['bool|int', 'X[bool, "bar"]|X[int, "foo"]']

    # first has a string, others don't
    samples = [
        (ti('int'), ti('X', args=(ti('int'), '"foo"'))),
        (ti('bool'), ti('X', args=(ti('bool'), ti('bool')))),
    ]
    assert generalize(samples) == ['bool|int', 'X[bool, bool]|X[int, "foo"]']

    samples = [
        (ti('X', args=(ti('int'), '"foo"')),),
        (ti('X', args=(ti('bool'), '"bar"')),),
    ]
    assert generalize(samples) == ['X[bool, "bar"]|X[int, "foo"]']


def test_is_typevar():
    assert not TypeInfo.from_type(int).is_typevar()
    assert not TypeInfo.from_set(TypeInfoSet((
        TypeInfo("", "list", args=(
            TypeInfo.from_type(int),
        )),
        TypeInfo.from_type(int),
        TypeInfo.from_type(bool),
    ))).is_typevar()

    assert TypeInfo.from_set(
        TypeInfoSet((
            TypeInfo.from_type(int),
            TypeInfo.from_type(bool),
        )),
        typevar_index=1
    ).is_typevar()

    assert TypeInfo("", "list", args=(
        TypeInfo.from_set(
            TypeInfoSet((
                TypeInfo.from_type(int),
                TypeInfo.from_type(bool),
            )),
            typevar_index=1
        ),
    )).is_typevar()
