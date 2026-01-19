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


# =============================================================================
# Tests for iterable generalization (ABC-based)
# =============================================================================

import collections.abc as abc


def test_find_common_abc():
    """ABCFinder.find_common_abc should return most specific shared ABC."""
    from righttyper.type_id import ABCFinder

    # list and tuple share Sequence
    common = ABCFinder.find_common_abc([list, tuple])
    assert common == abc.Sequence

    # list and set share Collection (not Sequence)
    common = ABCFinder.find_common_abc([list, set])
    assert common == abc.Collection

    # list and dict share Collection (dict is also a Collection)
    common = ABCFinder.find_common_abc([list, dict])
    assert common == abc.Collection

    # Empty list returns None
    assert ABCFinder.find_common_abc([]) is None

    # Single type returns its most specific ABC
    common = ABCFinder.find_common_abc([list])
    assert common is not None


def test_generalize_iterables_to_common_abc():
    """Multiple iterables should generalize to their common ABC."""
    # list, tuple, range are all Sequences
    types = {
        tt(list, args=(tt(int),)),
        tt(tuple, args=(tt(int),)),
        tt(range),
    }
    result = righttyper.generalize.merged_types(types)
    # Should become Sequence (the common ABC for list, tuple, range)
    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert result.type_obj == abc.Sequence, f"Expected Sequence, got {result.type_obj}"


def test_generalize_iterables_with_element_type():
    """Iterables with known element types should preserve them."""
    # list[int] and tuple[int] - both have int element type
    types = {
        tt(list, args=(tt(int),)),
        tt(tuple, args=(tt(int),)),
    }
    result = righttyper.generalize.merged_types(types)
    # Should become Sequence[int]
    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert result.type_obj == abc.Sequence, f"Expected Sequence, got {result.type_obj}"
    assert len(result.args) == 1, f"Expected 1 type arg, got {len(result.args)}"
    assert result.args[0].type_obj == int, f"Expected int element, got {result.args[0]}"


def test_generalize_iterables_mixed_element_types():
    """Iterables with different element types should first merge via merge_similar_generics."""
    types = {
        tt(list, args=(tt(int),)),
        tt(list, args=(tt(str),)),
    }
    result = righttyper.generalize.merged_types(types, for_variable=True)
    # merge_similar_generics should handle this: list[int]|list[str] -> list[int|str]
    assert result.name == 'list'
    assert result.args  # has element type


def test_generalize_iterables_unknown_element():
    """Iterables without element type info should use bare ABC."""
    types = {
        tt(range),  # no element type recorded
        tt(list, args=(tt(int),)),
    }
    result = righttyper.generalize.merged_types(types)
    # Should become Sequence (bare, no element type since range has none)
    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert result.type_obj == abc.Sequence, f"Expected Sequence, got {result.type_obj}"
    # No element type args since range doesn't have one
    assert len(result.args) == 0, f"Expected no type args (bare ABC), got {result.args}"


def test_generalize_iterables_excludes_str_bytes():
    """str and bytes should not be grouped with other iterables."""
    types = {
        tt(str),
        tt(list, args=(tt(int),)),
    }
    result = righttyper.generalize.merged_types(types)
    # Should remain as union since str is not grouped with list
    assert result.is_union()
    assert len(result.args) == 2


def test_generalize_iterables_excludes_bytes():
    """bytes should not be grouped with other iterables."""
    types = {
        tt(bytes),
        tt(list, args=(tt(int),)),
    }
    result = righttyper.generalize.merged_types(types)
    # Should remain as union since bytes is not grouped with list
    assert result.is_union()
    assert len(result.args) == 2


def test_generalize_mixed_union():
    """Only iterable subset should be generalized, others kept as-is."""
    # list[int] | str | None | range
    types = {
        tt(list, args=(tt(int),)),
        tt(str),
        TypeInfo.from_type(type(None)),
        tt(range),
    }
    result = righttyper.generalize.merged_types(types)

    # Should have: Sequence (from list+range), str, None = 3 members
    # NOT: list[int]|range|str|None (4 members)
    assert result.is_union(), f"Expected union, got {result}"
    assert len(result.args) == 3, f"Expected 3 members (Sequence, str, None), got {len(result.args)}: {result}"

    # Check that one of the members is Sequence
    type_objs = [t.type_obj for t in result.args if isinstance(t, TypeInfo)]
    assert abc.Sequence in type_objs, f"Expected Sequence in union, got {type_objs}"
    assert str in type_objs, f"Expected str in union, got {type_objs}"


def test_generalize_single_iterable_no_change():
    """A single iterable should not be generalized."""
    types = {
        tt(list, args=(tt(int),)),
        tt(str),
    }
    result = righttyper.generalize.merged_types(types)
    # list is the only "real" iterable (str excluded), so no generalization
    assert result.is_union()


def test_generalize_mappings():
    """Multiple mappings with same container should merge via merge_similar_generics."""
    types = {
        tt(dict, args=(tt(str), tt(int))),
        tt(dict, args=(tt(str), tt(str))),
    }
    result = righttyper.generalize.merged_types(types, for_variable=True)
    # merge_similar_generics handles: dict[str, int]|dict[str, str] -> dict[str, int|str]
    assert result.name == 'dict'
    assert not result.is_union()
