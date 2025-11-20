from righttyper.typeinfo import TypeInfo
from righttyper.type_id import normalize_module_name
import righttyper.generalize
from typing import Any, Never, Self
import pytest
from righttyper.options import output_options


def ti(name: str, **kwargs) -> TypeInfo:
    return TypeInfo(module='', name=name, **kwargs)


def generalize(samples):
    result = righttyper.generalize.generalize(samples)
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


def test_no_simplify_types(monkeypatch):
    monkeypatch.setattr(output_options, 'simplify_types', False)

    # in Python, issubclass(bool, int)
    both = {TypeInfo.from_type(int), TypeInfo.from_type(bool)}
    assert righttyper.generalize.merged_types(both) == TypeInfo.from_set(both)
    assert len(TypeInfo.from_set(both).args) == 2


def test_generalize_empty():
    assert generalize([]) == []
    assert generalize([tuple()]) == []


def test_generalize_single_sample():
    assert generalize([(ti('int'), ti('float'), ti('str'))]) == ['int', 'float', 'str']
    assert generalize([(ti('int'), ti('int'), ti('int'))]) == ['int', 'int', 'int']


def test_generalize_varied_length_samples():
    samples = [
        (ti('int'), ti('int')),
        (ti('int'), ti('int'), ti('int')),
    ]
    # TODO raise exception instead?
    assert generalize(samples) is None


def test_generalize_uniform_single_type():
    samples = [
        (ti('int'), ti('int'), ti('int')),
        (ti('bool'), ti('bool'), ti('bool')),
        (ti('float'), ti('float'), ti('float'))
    ]
    assert generalize(samples) == ['T1', 'T1', 'T1']


def test_generalize_uniform_single_type_with_generic():
    samples = [
        (ti('int'), ti('int')),
        (ti('bool'), ti('bool')),
        (ti('X', args=(ti('foo'),)), ti('X', args=(ti('foo'),))),
    ]

    assert generalize(samples) == ['T1', 'T1']


def test_generalize_first_same_then_different():
    samples = [
        (ti('int'), ti('int')),
        (ti('bool'), ti('bool')),
        (ti('int'), ti('bool'))
    ]
    assert generalize(samples) == ['bool|int', 'bool|int']


def test_generalize_mixed_with_constant_types():
    samples = [
        (ti('int'), ti('str'), ti('int')),
        (ti('bool'), ti('str'), ti('float')),
        (ti('float'), ti('str'), ti('bool'))
    ]
    assert generalize(samples) == ['bool|float|int', 'str', 'bool|float|int']


def test_generalize_shared_variability():
    samples = [
        (ti('int'), ti('int'), ti('bool'), ti('int')),
        (ti('float'), ti('float'), ti('bool'), ti('float'))
    ]
    assert generalize(samples) == ['T1', 'T1', 'bool', 'T1']


def test_generalize_all_distinct_types():
    samples = [
        (ti('int'), ti('str'), ti('float'), ti('bool')),
        (ti('float'), ti('str'), ti('bool'), ti('int'))
    ]
    assert generalize(samples) == ['float|int', 'str', 'bool|float', 'bool|int']


def test_generalize_generic():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
    ]
    assert generalize(samples) == ['T1', 'list[T1]']


def test_generalize_generic_not_generalizable():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
        (ti('bool'), TypeInfo.from_type(type(None))),
    ]
    assert generalize(samples) == ['bool|float|int', 'list[float]|list[int]|None']

    samples = [
        (ti('int'), ti('tuple', args=(ti('int'),))),
        (ti('float'), ti('tuple', args=(ti('float'),))),
        (ti('bool'), ti('tuple', args=(ti('bool'), ti('int'))))
    ]
    assert generalize(samples) == ['bool|float|int', 'tuple[bool, int]|tuple[float]|tuple[int]']


def test_generalize_generic_among_options():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('X', args=(ti('foo'),)), ti('list', args=(ti('X', args=(ti('foo'),)),))),
    ]
    assert generalize(samples) == ['T1', 'list[T1]']


def test_generalize_generic_within_args():
    samples = [
        (ti('tuple', args=(ti('int'),)), ti('list', args=(ti('int'),))),
        (ti('tuple', args=(ti('float'),)), ti('list', args=(ti('float'),))),
    ]
    assert generalize(samples) == ['tuple[T1]', 'list[T1]']


def test_generalize_with_ellipsis():
    samples = [
        (ti('int'), ti('tuple', args=(ti('int'), ...))),
        (ti('float'), ti('tuple', args=(ti('float'), ...))),
    ]
    assert generalize(samples) == ['T1', 'tuple[T1, ...]']


def test_generalize_callable():
    samples: list[tuple[TypeInfo, ...]] = [
        (ti('Callable', args=(TypeInfo.list([ti('int'), ti('int')]), ti('int'))),),
        (ti('Callable', args=(TypeInfo.list([ti('str'), ti('str')]), ti('str'))),),
    ]
    assert generalize(samples) == ['Callable[[T1, T1], T1]']

    samples = [
        (ti('int'), ti('Callable', args=(..., ti('int'))),),
        (ti('str'), ti('Callable', args=(..., ti('str'))),),
    ]
    assert generalize(samples) == ['T1', 'Callable[..., T1]']


def test_generalize_generic_with_string():
    samples: Any = [
        (ti('int'), ti('X', args=(ti('int'), 'foo'))),
        (ti('bool'), ti('X', args=(ti('bool'), 'bar'))),
    ]
    assert generalize(samples) == ['bool|int', 'X[bool, "bar"]|X[int, "foo"]']

    # first has a string, others don't
    samples = [
        (ti('int'), ti('X', args=(ti('int'), 'foo'))),
        (ti('bool'), ti('X', args=(ti('bool'), ti('bool')))),
    ]
    assert generalize(samples) == ['bool|int', 'X[bool, bool]|X[int, "foo"]']

    samples = [
        (ti('X', args=(ti('int'), 'foo')),),
        (ti('X', args=(ti('bool'), 'bar')),),
    ]
    assert generalize(samples) == ['X[bool, "bar"]|X[int, "foo"]']


def test_generalize_jaxtyping_dimensions():
    samples = [
        (
            TypeInfo('', 'int'),
            TypeInfo('jaxtyping', 'Float64', args=(
                    TypeInfo('np', 'ndarray'),
                    '10 20'
                )
            ),
            TypeInfo('jaxtyping', 'Float64', args=(
                    TypeInfo('np', 'ndarray'),
                    '20'
                )
            )
        ),
        (
            TypeInfo('', 'int'),
            TypeInfo('jaxtyping', 'Float64', args=(
                    TypeInfo('np', 'ndarray'),
                    '10 10'
                )
            ),
            TypeInfo('jaxtyping', 'Float64', args=(
                    TypeInfo('np', 'ndarray'),
                    '10'
                )
            )
        )
    ]
    assert generalize(samples) == [
        'int', 'jaxtyping.Float64[np.ndarray, "10 D1"]', 'jaxtyping.Float64[np.ndarray, "D1"]'
    ]


def test_generalize_jaxtyping_single_sample():
    samples = [
        (
            TypeInfo('', 'int'),
            TypeInfo('jaxtyping', 'Float64', args=(
                    TypeInfo('np', 'ndarray'),
                    '10 20'
                )
            ),
            TypeInfo('jaxtyping', 'Float64', args=(
                    TypeInfo('np', 'ndarray'),
                    '20'
                )
            )
        ),
    ]
    assert generalize(samples) == [
        'int', 'jaxtyping.Float64[np.ndarray, "10 20"]', 'jaxtyping.Float64[np.ndarray, "20"]'
    ]


def tt(t, **kwargs) -> TypeInfo:
    return TypeInfo(
        name=t.__qualname__,
        module=normalize_module_name(t.__module__),
        type_obj=t,
        **kwargs
    )


def test_generalize_type_and_never():
    samples = [
        (tt(dict, args=(tt(str), tt(str))), tt(Self)),
        (tt(dict, args=(tt(Never), tt(Never))), tt(Self)),
    ]
    assert generalize(samples) == ['dict[str, str]', 'typing.Self']
