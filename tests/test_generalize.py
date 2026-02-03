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


# Tests for container superset merging

def test_merge_container_supersets_list():
    """list[int] | list[int|str] -> list[int|str] (int ⊆ int|str)"""
    from righttyper.generalize import merge_container_supersets

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_int_str = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))

    result = merge_container_supersets({list_int, list_int_str})
    assert result == {list_int_str}


def test_merge_container_supersets_no_subset():
    """list[int] | list[str] -> unchanged (no subset relationship)"""
    from righttyper.generalize import merge_container_supersets

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_str = TypeInfo.from_type(list).replace(args=(str_t,))

    result = merge_container_supersets({list_int, list_str})
    assert result == {list_int, list_str}


def test_merge_container_supersets_chain():
    """list[int] | list[int|str] | list[int|str|float] -> list[int|str|float]"""
    from righttyper.generalize import merge_container_supersets

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    float_t = TypeInfo.from_type(float)

    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_int_str = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))
    list_all = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t, float_t}),))

    result = merge_container_supersets({list_int, list_int_str, list_all})
    assert result == {list_all}


def test_merge_container_supersets_nested():
    """list[set[int]] | list[set[int|str]] -> list[set[int|str]]"""
    from righttyper.generalize import merge_container_supersets

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    set_int = TypeInfo.from_type(set).replace(args=(int_t,))
    set_int_str = TypeInfo.from_type(set).replace(args=(TypeInfo.from_set({int_t, str_t}),))
    list_set_int = TypeInfo.from_type(list).replace(args=(set_int,))
    list_set_int_str = TypeInfo.from_type(list).replace(args=(set_int_str,))

    result = merge_container_supersets({list_set_int, list_set_int_str})
    assert result == {list_set_int_str}


def test_merge_container_supersets_dict():
    """dict[str, int] | dict[str, int|float] -> dict[str, int|float]"""
    from righttyper.generalize import merge_container_supersets

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)
    float_t = TypeInfo.from_type(float)

    dict_str_int = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    dict_str_int_float = TypeInfo.from_type(dict).replace(args=(str_t, TypeInfo.from_set({int_t, float_t})))

    result = merge_container_supersets({dict_str_int, dict_str_int_float})
    assert result == {dict_str_int_float}


def test_merge_container_supersets_dict_no_subset():
    """dict[str, int] | dict[int, int] -> unchanged (no subset relationship in keys)"""
    from righttyper.generalize import merge_container_supersets

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)

    dict_str_int = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    dict_int_int = TypeInfo.from_type(dict).replace(args=(int_t, int_t))

    result = merge_container_supersets({dict_str_int, dict_int_int})
    assert result == {dict_str_int, dict_int_int}


def test_merge_container_supersets_dict_key_subset():
    """dict[str, int] | dict[str|int, int] -> dict[str|int, int] (str ⊆ str|int)"""
    from righttyper.generalize import merge_container_supersets

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)

    dict_str_int = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    dict_str_or_int_int = TypeInfo.from_type(dict).replace(args=(TypeInfo.from_set({str_t, int_t}), int_t))

    result = merge_container_supersets({dict_str_int, dict_str_or_int_int})
    assert result == {dict_str_or_int_int}


def test_merge_container_supersets_mixed_containers():
    """list[int] | set[int] | list[int|str] -> set[int] | list[int|str]"""
    from righttyper.generalize import merge_container_supersets

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_int_str = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))
    set_int = TypeInfo.from_type(set).replace(args=(int_t,))

    result = merge_container_supersets({list_int, set_int, list_int_str})
    assert result == {set_int, list_int_str}


# Tests for covariant type merging in merged_types

def test_merge_covariant_tuple():
    """tuple[int] | tuple[bool] -> tuple[int] for params (covariant, bool <: int)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    bool_t = TypeInfo.from_type(bool)
    tuple_int = TypeInfo.from_type(tuple).replace(args=(int_t,))
    tuple_bool = TypeInfo.from_type(tuple).replace(args=(bool_t,))

    result = merged_types({tuple_int, tuple_bool}, for_variable=False)
    assert result == TypeInfo.from_type(tuple).replace(args=(int_t,))


def test_merge_covariant_tuple_multi_arg():
    """tuple[int, str] | tuple[float, str] -> tuple[float, str] for params
    (int simplifies into float via numeric tower)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    float_t = TypeInfo.from_type(float)
    str_t = TypeInfo.from_type(str)
    tuple_is = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))
    tuple_fs = TypeInfo.from_type(tuple).replace(args=(float_t, str_t))

    result = merged_types({tuple_is, tuple_fs}, for_variable=False)
    # int|float simplifies to float via PEP 3141 numeric tower
    expected = TypeInfo.from_type(tuple).replace(args=(float_t, str_t))
    assert result == expected


def test_merge_covariant_tuple_varlen():
    """tuple[int, ...] | tuple[str, ...] -> tuple[int|str, ...] for params"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    tuple_int_var = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))
    tuple_str_var = TypeInfo.from_type(tuple).replace(args=(str_t, Ellipsis))

    result = merged_types({tuple_int_var, tuple_str_var}, for_variable=False)
    expected = TypeInfo.from_type(tuple).replace(
        args=(TypeInfo.from_set({int_t, str_t}), Ellipsis)
    )
    assert result == expected


def test_merge_covariant_frozenset():
    """frozenset[int] | frozenset[str] -> frozenset[int|str] for params"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    fs_int = TypeInfo.from_type(frozenset).replace(args=(int_t,))
    fs_str = TypeInfo.from_type(frozenset).replace(args=(str_t,))

    result = merged_types({fs_int, fs_str}, for_variable=False)
    expected = TypeInfo.from_type(frozenset).replace(
        args=(TypeInfo.from_set({int_t, str_t}),)
    )
    assert result == expected


def test_merge_covariant_tuple_different_lengths():
    """tuple[int] | tuple[int, str] -> unchanged (different arities)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    tuple_1 = TypeInfo.from_type(tuple).replace(args=(int_t,))
    tuple_2 = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))

    result = merged_types({tuple_1, tuple_2}, for_variable=False)
    assert result == TypeInfo.from_set({tuple_1, tuple_2})


def test_merge_covariant_tuple_fixed_vs_varlen():
    """tuple[int, str] | tuple[int, ...] -> unchanged (structural mismatch)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    tuple_fixed = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))
    tuple_var = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))

    result = merged_types({tuple_fixed, tuple_var}, for_variable=False)
    assert result == TypeInfo.from_set({tuple_fixed, tuple_var})


def test_merge_invariant_list_unchanged():
    """list[int] | list[str] -> unchanged for params (lists are invariant)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_str = TypeInfo.from_type(list).replace(args=(str_t,))

    result = merged_types({list_int, list_str}, for_variable=False)
    assert result == TypeInfo.from_set({list_int, list_str})


# Tests for fixed-length tuple subsumption by variable-length tuples

def test_subsume_fixed_by_varlen_tuple():
    """tuple[int, int] | tuple[int, ...] -> tuple[int, ...] (all elements match)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    tuple_fixed = TypeInfo.from_type(tuple).replace(args=(int_t, int_t))
    tuple_var = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))

    result = merged_types({tuple_fixed, tuple_var}, for_variable=False)
    assert result == tuple_var


def test_no_subsume_fixed_by_varlen_tuple():
    """tuple[int, str] | tuple[int, ...] -> unchanged (str not contained by int)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    tuple_fixed = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))
    tuple_var = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))

    result = merged_types({tuple_fixed, tuple_var}, for_variable=False)
    assert result == TypeInfo.from_set({tuple_fixed, tuple_var})


def test_subsume_fixed_by_varlen_tuple_union_elem():
    """tuple[int, str] | tuple[int|str, ...] -> tuple[int|str, ...] (all elements in union)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    int_or_str = TypeInfo.from_set({int_t, str_t})
    tuple_fixed = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))
    tuple_var = TypeInfo.from_type(tuple).replace(args=(int_or_str, Ellipsis))

    result = merged_types({tuple_fixed, tuple_var}, for_variable=False)
    assert result == tuple_var


def test_subsume_multiple_fixed_by_varlen():
    """tuple[int] | tuple[int, int] | tuple[int, ...] -> tuple[int, ...]"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    tuple_1 = TypeInfo.from_type(tuple).replace(args=(int_t,))
    tuple_2 = TypeInfo.from_type(tuple).replace(args=(int_t, int_t))
    tuple_var = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))

    result = merged_types({tuple_1, tuple_2, tuple_var}, for_variable=False)
    assert result == tuple_var


def test_subsume_empty_tuple_by_varlen():
    """tuple[()] | tuple[int, ...] -> tuple[int, ...] (empty tuple is zero-or-more ints)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    tuple_empty = TypeInfo.from_type(tuple).replace(args=((),))
    tuple_var = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))

    result = merged_types({tuple_empty, tuple_var}, for_variable=False)
    assert result == tuple_var
