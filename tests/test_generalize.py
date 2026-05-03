from righttyper.typeinfo import TypeInfo, TypeInfoArg
from righttyper.type_id import normalize_module_name
import righttyper.generalize
from typing import Any, Never, Self, cast
import pytest
from righttyper.options import output_options


def ti(name: str, **kwargs) -> TypeInfo:
    return TypeInfo(module='', name=name, **kwargs)


def _ti(a: TypeInfoArg) -> TypeInfo:
    """Narrow a TypeInfoArg to TypeInfo for tests where it's known to be one."""
    assert isinstance(a, TypeInfo)
    return a


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


def test_generalize_callable_different_code_ids():
    """Resolved Callables with different code_ids and different signatures
    should produce a union, not be merged as homogeneous."""
    from righttyper.righttyper_types import CodeId, Filename, FunctionName

    code_id_a = CodeId(Filename('f.py'), FunctionName('add_numbers'), 10, 0)
    code_id_b = CodeId(Filename('f.py'), FunctionName('greet'), 20, 0)

    # After ResolvingT, Callable types have actual args and retain their code_ids
    samples: list[tuple[TypeInfo, ...]] = [
        (ti('Callable', args=(TypeInfo.list([ti('int'), ti('int')]), ti('int')), code_id=code_id_a),),
        (ti('Callable', args=(TypeInfo.list([ti('str')]), ti('str')), code_id=code_id_b),),
    ]
    result = generalize(samples)
    assert result is not None
    assert len(result) == 1
    # Should produce a union of complete Callable types, not merge their internals
    assert result == ['Callable[[int, int], int]|Callable[[str], str]']


def test_generalize_clears_nested_typevar_index():
    """When forming a union, nested typevar_index should be cleared.

    Types from resolved functions may have typevar_index set. When these types
    become part of a union (non-homogeneous), their typevar_index should be
    cleared to avoid spurious typevars in the output.
    """
    from righttyper.righttyper_types import CodeId, Filename, FunctionName

    code_id_a = CodeId(Filename('f.py'), FunctionName('add_numbers'), 10, 0)
    code_id_b = CodeId(Filename('f.py'), FunctionName('greet'), 20, 0)

    # Simulate resolved Callables where int|float has typevar_index=1 from add_numbers
    int_or_float = ti('int|float', typevar_index=1)
    samples: list[tuple[TypeInfo, ...]] = [
        (ti('Callable', args=(TypeInfo.list([int_or_float, int_or_float]), int_or_float), code_id=code_id_a),),
        (ti('Callable', args=(TypeInfo.list([ti('str')]), ti('str')), code_id=code_id_b),),
    ]
    result = generalize(samples)
    assert result is not None
    # The nested typevar_index should NOT produce a T1 in the output
    assert result == ['Callable[[int|float, int|float], int|float]|Callable[[str], str]']


def test_generalize_callable_no_code_id_same_arity():
    """Callables without code_ids and same param arity should decompose normally."""
    samples: list[tuple[TypeInfo, ...]] = [
        (ti('Callable', args=(TypeInfo.list([ti('int'), ti('int')]), ti('int'))),),
        (ti('Callable', args=(TypeInfo.list([ti('str'), ti('str')]), ti('str'))),),
    ]
    # Same arity: valid decomposition into typevars
    assert generalize(samples) == ['Callable[[T1, T1], T1]']


def test_generalize_callable_no_code_id_different_arity():
    """Callables without code_ids and different param arities should union, not decompose."""
    samples: list[tuple[TypeInfo, ...]] = [
        (ti('Callable', args=(TypeInfo.list([ti('int'), ti('int')]), ti('int'))),),
        (ti('Callable', args=(TypeInfo.list([ti('str')]), ti('str'))),),
    ]
    # Different arity: should produce a union, not try to decompose
    assert generalize(samples) == ['Callable[[int, int], int]|Callable[[str], str]']


def test_generalize_callable_list_type_info_never_typevar():
    """Parameter lists should never become typevars, even when the pattern repeats."""
    samples: list[tuple[TypeInfo, ...]] = [
        (ti('Callable', args=(TypeInfo.list([ti('int'), ti('int')]), ti('int'))),
         ti('Callable', args=(TypeInfo.list([ti('int'), ti('int')]), ti('int')))),
        (ti('Callable', args=(TypeInfo.list([ti('str')]), ti('str'))),
         ti('Callable', args=(TypeInfo.list([ti('str')]), ti('str')))),
    ]
    result = generalize(samples)
    assert result is not None
    # The whole Callable union repeats across both positions, so it becomes a typevar.
    # Crucially, the ListTypeInfo (param list) does NOT become its own typevar.
    assert result == ['T1', 'T1']


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
    """list[int] | list[int|str] -> list[int|str] for variables"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_int_str = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))

    result = merged_types({list_int, list_int_str}, for_variable=True)
    assert result == list_int_str


def test_merge_container_supersets_no_subset():
    """list[int] | list[str] -> list[int|str] for variables (merges args)"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_str = TypeInfo.from_type(list).replace(args=(str_t,))

    result = merged_types({list_int, list_str}, for_variable=True)
    assert result == TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))


def test_merge_container_supersets_chain():
    """list[int] | list[int|str] | list[int|str|float] -> list[int|str|float] for variables"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    float_t = TypeInfo.from_type(float)

    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_int_str = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))
    list_all = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t, float_t}),))

    result = merged_types({list_int, list_int_str, list_all}, for_variable=True)
    assert result == list_all


def test_merge_container_supersets_nested():
    """list[set[int]] | list[set[int|str]] -> list[set[int|str]] for variables"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    set_int = TypeInfo.from_type(set).replace(args=(int_t,))
    set_int_str = TypeInfo.from_type(set).replace(args=(TypeInfo.from_set({int_t, str_t}),))
    list_set_int = TypeInfo.from_type(list).replace(args=(set_int,))
    list_set_int_str = TypeInfo.from_type(list).replace(args=(set_int_str,))

    result = merged_types({list_set_int, list_set_int_str}, for_variable=True)
    assert result == list_set_int_str


def test_merge_container_supersets_dict():
    """dict[str, int] | dict[str, int|float] -> dict[str, int|float] for variables"""
    from righttyper.generalize import merged_types

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)
    float_t = TypeInfo.from_type(float)

    dict_str_int = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    dict_str_int_float = TypeInfo.from_type(dict).replace(args=(str_t, TypeInfo.from_set({int_t, float_t})))

    result = merged_types({dict_str_int, dict_str_int_float}, for_variable=True)
    assert result == dict_str_int_float


def test_merge_container_supersets_dict_no_subset():
    """dict[str, int] | dict[int, int] -> dict[str|int, int] for variables (merges keys)"""
    from righttyper.generalize import merged_types

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)

    dict_str_int = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    dict_int_int = TypeInfo.from_type(dict).replace(args=(int_t, int_t))

    result = merged_types({dict_str_int, dict_int_int}, for_variable=True)
    assert result == TypeInfo.from_type(dict).replace(args=(TypeInfo.from_set({str_t, int_t}), int_t))


def test_merge_container_supersets_dict_key_subset():
    """dict[str, int] | dict[str|int, int] -> dict[str|int, int] for variables"""
    from righttyper.generalize import merged_types

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)

    dict_str_int = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    dict_str_or_int_int = TypeInfo.from_type(dict).replace(args=(TypeInfo.from_set({str_t, int_t}), int_t))

    result = merged_types({dict_str_int, dict_str_or_int_int}, for_variable=True)
    assert result == dict_str_or_int_int


def test_merge_container_supersets_mixed_containers():
    """list[int] | set[int] | list[int|str] -> set[int] | list[int|str] for variables"""
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    list_int_str = TypeInfo.from_type(list).replace(args=(TypeInfo.from_set({int_t, str_t}),))
    set_int = TypeInfo.from_type(set).replace(args=(int_t,))

    result = merged_types({list_int, set_int, list_int_str}, for_variable=True)
    assert result == TypeInfo.from_set({set_int, list_int_str})


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
    """tuple[int] | tuple[int, str] -> tuple[int|str, ...] (different arities → varlen).

    This is a valid LUB but wider than the original union: it loses positional
    type information and length constraints. Acceptable for runtime-inferred
    annotations where observing multiple lengths implies variable-length usage.
    """
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    tuple_1 = TypeInfo.from_type(tuple).replace(args=(int_t,))
    tuple_2 = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))

    result = merged_types({tuple_1, tuple_2}, for_variable=False)
    assert result == TypeInfo.from_type(tuple).replace(
        args=(TypeInfo.from_set({int_t, str_t}), Ellipsis)
    )


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


# =============================================================================
# Tests for cross-container simplification
# =============================================================================

def test_list_and_empty_tuple_to_sequence():
    """list[tuple[int, int]] | tuple[()] → Sequence[tuple[int, int]].
    An empty tuple is an empty sequence, compatible with any element type."""
    import collections.abc
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    inner = TypeInfo.from_type(tuple).replace(args=(int_t, int_t))
    list_of_tuples = TypeInfo.from_type(list).replace(args=(inner,))
    empty_tuple = TypeInfo.from_type(tuple).replace(args=((),))

    result = merged_types({list_of_tuples, empty_tuple},
                          accessed_attributes={"__iter__"})

    # Should merge to Sequence[tuple[int, int]], not stay as a union
    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert result.type_obj is not None
    assert issubclass(result.type_obj, collections.abc.Sequence)
    # Element type should be preserved
    assert result.args and isinstance(result.args[0], TypeInfo)
    assert result.args[0] == inner


def test_list_and_varlen_tuple_to_sequence():
    """list[int] | tuple[int, ...] → Sequence[int] when accessed via sequence attrs.
    Two non-empty containers with compatible element types merge to a common ABC."""
    import collections.abc
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    list_of_int = TypeInfo.from_type(list).replace(args=(int_t,))
    tuple_of_int = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))

    result = merged_types({list_of_int, tuple_of_int},
                          accessed_attributes={"__iter__"})

    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Sequence)
    # Element type preserved
    assert result.args and isinstance(result.args[0], TypeInfo)
    assert _ti(result.args[0]).type_obj is int


def test_set_and_frozenset_to_abc_set():
    """set[int] | frozenset[int] → Set[int]."""
    import collections.abc
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    s = TypeInfo.from_type(set).replace(args=(int_t,))
    fs = TypeInfo.from_type(frozenset).replace(args=(int_t,))

    result = merged_types({s, fs}, accessed_attributes={"__iter__"})
    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Set)


def test_dict_and_ordered_dict_to_mapping():
    """dict[str, int] | OrderedDict[str, int] → Mapping[str, int] (2 type args)."""
    import collections
    import collections.abc
    from righttyper.generalize import merged_types

    str_t = TypeInfo.from_type(str)
    int_t = TypeInfo.from_type(int)
    d = TypeInfo.from_type(dict).replace(args=(str_t, int_t))
    od = TypeInfo.from_type(collections.OrderedDict).replace(args=(str_t, int_t))

    result = merged_types({d, od}, accessed_attributes={"items"})
    assert not result.is_union(), f"Expected single type, got union: {result}"
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Mapping)
    # Both key and value types preserved
    assert len(result.args) == 2
    assert result.args[0] == str_t
    assert result.args[1] == int_t


def test_list_and_empty_tuple_without_accessed_attrs():
    """list[X] | tuple[()] → Sequence[X] even without accessed_attributes,
    since an empty tuple is compatible with any element type."""
    import collections.abc as abc
    from righttyper.generalize import merged_types

    int_t = TypeInfo.from_type(int)
    inner = TypeInfo.from_type(tuple).replace(args=(int_t, int_t))
    list_of_tuples = TypeInfo.from_type(list).replace(args=(inner,))
    empty_tuple = TypeInfo.from_type(tuple).replace(args=((),))

    result = merged_types({list_of_tuples, empty_tuple})
    assert not result.is_union()
    assert result.type_obj is abc.Sequence
    assert result.args[0] == inner


# =============================================================================
# Tests for attribute-aware simplification
# =============================================================================

# Test hierarchy: Base has .name and .value; ChildA adds .extra_a; ChildB adds .extra_b.
# Without accessed_attributes, simplify merges ChildA|ChildB → Base (common dir() attrs).
# With accessed_attributes={'name'}, the merge should still work (Base has .name).
# With accessed_attributes={'extra_a'}, merge should NOT happen (Base lacks .extra_a).

class Base:
    name: str = ""
    value: int = 0

class ChildA(Base):
    extra_a: str = ""

class ChildB(Base):
    extra_b: str = ""


def test_lub_with_accessed_attributes():
    """When accessed_attributes are provided and the common base has them, merge happens."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(ChildA)
    b = TypeInfo.from_type(ChildB)

    result = merged_types({a, b}, accessed_attributes={"name"})
    assert not result.is_union()
    assert result.type_obj is Base


def test_lub_without_accessed_attributes():
    """Without accessed_attributes, dir()-based MRO merge to common base."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(ChildA)
    b = TypeInfo.from_type(ChildB)

    result = merged_types({a, b})
    assert not result.is_union()
    assert result.type_obj is Base


def test_lub_accessed_attrs_prevents_over_merge():
    """When accessed_attributes include an attr not on the common base, merge is prevented."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(ChildA)
    b = TypeInfo.from_type(ChildB)

    # extra_a is only on ChildA, not on Base → can't merge to Base
    result = merged_types({a, b}, accessed_attributes={"extra_a"})
    assert result.is_union()
    types = {t.type_obj for t in result.to_set()}
    assert types == {ChildA, ChildB}


def test_lub_single_type_with_accessed_attributes():
    """Even a single type can be generalized to a base when accessed_attributes
    are all present on that base."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(ChildA)

    # 'name' is on Base → ChildA can be generalized to Base
    result = merged_types({a}, accessed_attributes={"name"})
    assert result.type_obj is Base


def test_lub_single_type_no_generalization_without_attrs():
    """Without accessed_attributes, a single type stays as-is."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(ChildA)

    result = merged_types({a})
    assert result.type_obj is ChildA


# =============================================================================
# Tests for ABC/protocol matching in simplification
# =============================================================================

class _IterableA:
    """Implements Iterable (recognized by __subclasshook__)."""
    def __iter__(self): return iter(())
    def __len__(self): return 0

class _IterableB:
    """Different class, also implements Iterable."""
    def __iter__(self): return iter(())
    def __len__(self): return 0


def test_lub_abc_fallback():
    """When types share no concrete base (only object), falls back to
    ABC matching. _IterableA and _IterableB both implement Iterable via
    __subclasshook__, so accessing __iter__ should merge to Iterable."""
    import collections.abc as abc
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(_IterableA)
    b = TypeInfo.from_type(_IterableB)

    result = merged_types({a, b}, accessed_attributes={"__iter__"})
    assert not result.is_union()
    assert issubclass(cast(type, result.type_obj), abc.Iterable)


def test_lub_abc_not_used_when_concrete_base_exists():
    """When a concrete base exists and has the accessed attributes,
    prefer it over ABC matching."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(ChildA)
    b = TypeInfo.from_type(ChildB)

    # 'name' is on Base → concrete merge to Base, not to some ABC
    result = merged_types({a, b}, accessed_attributes={"name"})
    assert not result.is_union()
    assert result.type_obj is Base


def test_lub_abc_single_type():
    """A single type is not generalized to an ABC — ABC matching only
    reduces union size, not replaces a single concrete type."""
    from righttyper.generalize import merged_types

    a = TypeInfo.from_type(_IterableA)

    result = merged_types({a}, accessed_attributes={"__len__"})
    assert result.type_obj is _IterableA


# =============================================================================
# Tests for lub(a, b) — least upper bound
# =============================================================================

def test_lub_identity():
    """lub(a, a) = a."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(int)
    assert lub(a, a) == a


def test_lub_never():
    """lub(a, Never) = a."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(int)
    never = TypeInfo.from_type(Never)
    assert lub(a, never) == a
    assert lub(never, a) == a


def test_lub_any():
    """lub(a, Any) = Any."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(int)
    any_t = TypeInfo.from_type(Any)
    assert lub(a, any_t) == any_t
    assert lub(any_t, a) == any_t


def test_lub_subtype_bool_int():
    """lub(bool, int) = int (bool <: int)."""
    from righttyper.generalize import lub
    assert lub(TypeInfo.from_type(bool), TypeInfo.from_type(int)) == TypeInfo.from_type(int)
    assert lub(TypeInfo.from_type(int), TypeInfo.from_type(bool)) == TypeInfo.from_type(int)


def test_lub_numeric_tower():
    """lub(int, float) = float (numeric tower)."""
    from righttyper.generalize import lub
    assert lub(TypeInfo.from_type(int), TypeInfo.from_type(float)).type_obj is float


def test_lub_no_common_base():
    """lub(int, str) = int|str (no useful common supertype)."""
    from righttyper.generalize import lub
    result = lub(TypeInfo.from_type(int), TypeInfo.from_type(str))
    assert result.is_union()
    types = {t.type_obj for t in result.args if isinstance(t, TypeInfo)}
    assert types == {int, str}


def test_lub_same_container_merge_for_variable():
    """lub(list[int], list[str]) = list[int|str] when for_variable=True."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(list).replace(args=(int_t,))
    b = TypeInfo.from_type(list).replace(args=(str_t,))
    result = lub(a, b, for_variable=True)
    assert result.type_obj is list
    assert result.args and isinstance(result.args[0], TypeInfo)
    assert _ti(result.args[0]).is_union()


def test_lub_same_container_no_merge_invariant():
    """lub(list[int], list[str]) = list[int]|list[str] when for_variable=False (invariant)."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(list).replace(args=(int_t,))
    b = TypeInfo.from_type(list).replace(args=(str_t,))
    result = lub(a, b, for_variable=False)
    assert result.is_union()
    assert result.to_set() == {a, b}


def test_lub_same_container_subtype_args():
    """lub(list[bool], list[int]) = list[int] when for_variable=True (bool <: int)."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(bool),))
    b = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(int),))
    result = lub(a, b, for_variable=True)
    assert result.type_obj is list
    assert _ti(result.args[0]).type_obj is int


def test_lub_covariant_tuple_always_merges():
    """lub(tuple[int,...], tuple[str,...]) = tuple[int|str,...] even for_variable=False."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))
    b = TypeInfo.from_type(tuple).replace(args=(str_t, Ellipsis))
    result = lub(a, b, for_variable=False)
    assert result.type_obj is tuple
    assert _ti(result.args[0]).is_union()


def test_lub_varlen_subsumes_fixed():
    """lub(tuple[int,int], tuple[int,...]) = tuple[int,...]."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    fixed = TypeInfo.from_type(tuple).replace(args=(int_t, int_t))
    varlen = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))
    assert lub(fixed, varlen) == varlen
    assert lub(varlen, fixed) == varlen


def test_lub_commutative():
    """lub(a, b) == lub(b, a) for various type pairs."""
    from righttyper.generalize import lub
    pairs = [
        (TypeInfo.from_type(int), TypeInfo.from_type(str)),
        (TypeInfo.from_type(bool), TypeInfo.from_type(int)),
        (TypeInfo.from_type(int), TypeInfo.from_type(float)),
    ]
    for a, b in pairs:
        assert lub(a, b) == lub(b, a), f"lub not commutative for {a}, {b}"


def test_lub_idempotent():
    """lub(a, a) == a for various types."""
    from righttyper.generalize import lub
    for t in [int, str, float, list, dict]:
        a = TypeInfo.from_type(t)
        assert lub(a, a) == a


def test_lub_empty_tuple_and_list_to_sequence():
    """lub(tuple[()], list[int]) → Sequence[int] (cross-container, common ABC)."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    non_empty = TypeInfo.from_type(list).replace(args=(int_t,))
    result = lub(empty, non_empty)
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Sequence)
    assert result.args and _ti(result.args[0]).type_obj is int
    # Commutative
    assert lub(non_empty, empty) == result


def test_lub_empty_same_container():
    """lub(list[Never], list[int]) → list[int] (same container, empty subsumed)."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    empty = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(Never),))
    non_empty = TypeInfo.from_type(list).replace(args=(int_t,))
    assert lub(empty, non_empty) == non_empty
    assert lub(non_empty, empty) == non_empty


def test_lub_empty_dict_and_list_to_collection():
    """lub(dict[Never,Never], list[int]) → Collection[int] (both are Collections)."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    empty_dict = TypeInfo.from_type(dict).replace(args=(TypeInfo.from_type(Never), TypeInfo.from_type(Never)))
    list_int = TypeInfo.from_type(list).replace(args=(int_t,))
    result = lub(empty_dict, list_int)
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Collection)


def test_lub_empty_dict_and_defaultdict():
    """lub(dict[Never,Never], defaultdict[str, list[int]]) → dict[str, list[int]]
    (MRO common supertype with the non-empty container's args)."""
    import collections
    from righttyper.generalize import lub
    empty_dict = TypeInfo.from_type(dict, args=(TypeInfo.from_type(Never), TypeInfo.from_type(Never)))
    dd = TypeInfo.from_type(collections.defaultdict, args=(TypeInfo.from_type(str),
                            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),))))
    result = lub(empty_dict, dd)
    assert result == TypeInfo.from_type(dict, args=(TypeInfo.from_type(str),
                      TypeInfo.from_type(list, args=(TypeInfo.from_type(int),))))


def test_lub_empty_tuple_and_generator():
    """lub(tuple[()], Generator[str, None, None]) should reduce Generator to
    Iterator[str] first, then merge with the empty tuple → Iterable[str]."""
    import collections.abc
    from righttyper.generalize import lub
    empty_tuple = TypeInfo.from_type(tuple, args=((),))
    gen = TypeInfo.from_type(collections.abc.Generator, args=(
        TypeInfo.from_type(str),
        TypeInfo.from_type(type(None)),
        TypeInfo.from_type(type(None)),
    ))
    result = lub(empty_tuple, gen)
    assert result == TypeInfo.from_type(collections.abc.Iterable, args=(TypeInfo.from_type(str),))


def test_lub_generator_reduced_to_iterator_before_merge():
    """lub should reduce Generator[X, None, None] to Iterator[X] before
    merging, so 3 Generator args don't leak into the result."""
    import collections.abc
    from righttyper.generalize import lub
    gen = TypeInfo.from_type(collections.abc.Generator, args=(
        TypeInfo.from_type(int),
        TypeInfo.from_type(type(None)),
        TypeInfo.from_type(type(None)),
    ))
    lst = TypeInfo.from_type(list, args=(TypeInfo.from_type(int),))
    result = lub(gen, lst)
    assert result == TypeInfo.from_set({
        TypeInfo.from_type(collections.abc.Iterator, args=(TypeInfo.from_type(int),)),
        TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
    })


def test_lub_empty_tuple_subsumed_by_varlen():
    """lub(tuple[()], tuple[int, ...]) → tuple[int, ...] (empty subsumed by varlen)."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    varlen = TypeInfo.from_type(tuple).replace(args=(int_t, Ellipsis))
    assert lub(empty, varlen) == varlen
    assert lub(varlen, empty) == varlen


def test_lub_mro_common_base_with_attrs():
    """lub(ChildA, ChildB) → Base with accessed_attributes."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(ChildA)
    b = TypeInfo.from_type(ChildB)
    result = lub(a, b, accessed_attributes={"name"})
    assert not result.is_union()
    assert result.type_obj is Base


def test_lub_mro_common_base_without_attrs():
    """lub(ChildA, ChildB) → Base even without accessed_attributes (dir() check passes)."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(ChildA)
    b = TypeInfo.from_type(ChildB)
    result = lub(a, b)
    assert not result.is_union()
    assert result.type_obj is Base


def test_lub_mro_no_useful_base():
    """lub(int, str) stays as union (only 'object' in common)."""
    from righttyper.generalize import lub
    a, b = TypeInfo.from_type(int), TypeInfo.from_type(str)
    result = lub(a, b)
    assert result.is_union()
    assert result.to_set() == {a, b}


def test_lub_mro_skips_numeric_widening():
    """lub(int, float) uses numeric tower (rule 4), not MRO to complex."""
    from righttyper.generalize import lub
    result = lub(TypeInfo.from_type(int), TypeInfo.from_type(float))
    assert result.type_obj is float  # numeric tower, not complex


def test_lub_empty_tuple_and_fixed_tuple():
    """tuple[()]|tuple[str] should merge.
    An empty tuple + a fixed tuple of strings → variable-length tuple of strings."""
    from righttyper.generalize import lub
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    fixed = TypeInfo.from_type(tuple).replace(args=(TypeInfo.from_type(str),))
    result = lub(empty, fixed)
    # varlen tuple subsumes both
    assert result.type_obj is tuple
    assert not result.is_union()


def test_lub_list_and_empty_tuple_to_sequence():
    """list[tuple[int,int]]|tuple[()] → Sequence[tuple[int,int]]."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    inner = TypeInfo.from_type(tuple).replace(args=(int_t, int_t))
    list_t = TypeInfo.from_type(list).replace(args=(inner,))
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    result = lub(list_t, empty)
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Sequence)


def test_lub_set_and_empty_tuple_to_collection():
    """set[int]|tuple[()] → Collection[int] (both are Collections)."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    set_t = TypeInfo.from_type(set).replace(args=(int_t,))
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    result = lub(set_t, empty)
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Collection)
    assert result.args and _ti(result.args[0]).type_obj is int


def test_lub_list_str_and_empty_tuple_to_sequence():
    """list[str]|tuple[()] → Sequence[str]."""
    import collections.abc
    from righttyper.generalize import lub
    list_t = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(str),))
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    result = lub(list_t, empty)
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Sequence)
    assert result.args and _ti(result.args[0]).type_obj is str


def test_lub_list_and_tuple_same_elem_to_sequence():
    """list[str]|tuple[str] → Sequence[str] (cross-container, same element type)."""
    import collections.abc
    from righttyper.generalize import lub
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(list).replace(args=(str_t,))
    b = TypeInfo.from_type(tuple).replace(args=(str_t,))
    result = lub(a, b, accessed_attributes={"__iter__"})
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Sequence)
    assert result.args and _ti(result.args[0]).type_obj is str


def test_lub_list_and_tuple_different_elem():
    """list[int]|tuple[str] → Sequence[int|str] (immutable tuple makes it covariant)."""
    import collections.abc
    from righttyper.generalize import lub
    a = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(int),))
    b = TypeInfo.from_type(tuple).replace(args=(TypeInfo.from_type(str),))
    result = lub(a, b, accessed_attributes={"__iter__"})
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Sequence)
    assert result.args and _ti(result.args[0]).is_union()
    assert _ti(result.args[0]).to_set() == {TypeInfo.from_type(int), TypeInfo.from_type(str)}


def test_lub_dict_and_list_of_tuples():
    """dict[str,str]|list[tuple[str,str]] stays as union (incompatible arg arities)."""
    from righttyper.generalize import lub
    str_t = TypeInfo.from_type(str)
    d = TypeInfo.from_type(dict).replace(args=(str_t, str_t))
    lt = TypeInfo.from_type(list).replace(args=(
        TypeInfo.from_type(tuple).replace(args=(str_t, str_t)),
    ))
    result = lub(d, lt)
    assert result.is_union()
    assert result.to_set() == {d, lt}


def test_lub_subtype_narrowing():
    """lub(ChildA, Base) → Base (ChildA <: Base)."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(ChildA)
    base = TypeInfo.from_type(Base)
    assert lub(a, base) == base
    assert lub(base, a) == base


def test_lub_fixed_tuple_different_lengths_to_varlen():
    """tuple[int] | tuple[int, str] → tuple[int|str, ...] (from mypy's JoinSuite).
    Different-length fixed tuples merge to varlen with unioned element types."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(tuple).replace(args=(int_t,))
    b = TypeInfo.from_type(tuple).replace(args=(int_t, str_t))
    result = lub(a, b)
    assert result.type_obj is tuple
    assert not result.is_union()
    # Should be tuple[int|str, ...]
    assert len(result.args) == 2
    assert result.args[1] is Ellipsis


def test_lub_empty_and_fixed_tuple_to_varlen():
    """tuple[()] | tuple[int] → tuple[int, ...] (from mypy's JoinSuite: line 727)."""
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    fixed = TypeInfo.from_type(tuple).replace(args=(int_t,))
    result = lub(empty, fixed)
    assert result.type_obj is tuple
    assert not result.is_union()
    assert len(result.args) == 2
    assert result.args[1] is Ellipsis
    assert _ti(result.args[0]).type_obj is int


def test_lub_fixed_tuple_same_length_merge():
    """tuple[int, str] | tuple[float, str] → tuple[int|float, str] (covariant, same length)."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(tuple).replace(args=(TypeInfo.from_type(int), TypeInfo.from_type(str)))
    b = TypeInfo.from_type(tuple).replace(args=(TypeInfo.from_type(float), TypeInfo.from_type(str)))
    result = lub(a, b)
    assert result.type_obj is tuple
    assert not result.is_union()
    # First arg merged (int|float → float via numeric tower)
    assert _ti(result.args[0]).type_obj is float
    assert _ti(result.args[1]).type_obj is str


def test_lub_multiple_lists_merge_args():
    """list[int] | list[str] → list[int|str] when for_variable=True."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(int),))
    b = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(str),))
    result = lub(a, b, for_variable=True)
    assert result.type_obj is list
    assert _ti(result.args[0]).is_union()
    assert _ti(result.args[0]).to_set() == {TypeInfo.from_type(int), TypeInfo.from_type(str)}


def test_lub_abc_fallback_via_lub():
    """lub(IterableA, IterableB) → Iterable when no MRO base but both implement ABC."""
    import collections.abc
    from righttyper.generalize import lub

    a = TypeInfo.from_type(_IterableA)
    b = TypeInfo.from_type(_IterableB)

    # No common MRO base (only object), but both implement Iterable.
    # ABC matching only kicks in when it reduces a union (len > 1),
    # so we need accessed_attributes for the ABC pre-filter.
    result = lub(a, b, accessed_attributes={"__iter__"})
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Iterable)


def test_lub_abc_not_used_without_accessed_attrs():
    """Without accessed_attributes, ABC matching doesn't fire."""
    from righttyper.generalize import lub

    a = TypeInfo.from_type(_IterableA)
    b = TypeInfo.from_type(_IterableB)

    # No accessed_attributes → falls through to union
    result = lub(a, b)
    assert result.is_union()
    assert result.to_set() == {a, b}


def test_lub_abc_cross_container_same_args():
    """lub(list[int], set[int]) → Collection[int] via ABC when args match."""
    import collections.abc
    from righttyper.generalize import lub

    int_t = TypeInfo.from_type(int)
    a = TypeInfo.from_type(list).replace(args=(int_t,))
    b = TypeInfo.from_type(set).replace(args=(int_t,))

    result = lub(a, b, accessed_attributes={"__iter__"})
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Iterable)
    assert result.args and _ti(result.args[0]).type_obj is int


def test_lub_abc_cross_container_different_args_invariant():
    """lub(list[int], set[str]) stays as union when for_variable=False (invariant)."""
    from righttyper.generalize import lub

    a = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(int),))
    b = TypeInfo.from_type(set).replace(args=(TypeInfo.from_type(str),))

    result = lub(a, b, for_variable=False, accessed_attributes={"__iter__"})
    assert result.is_union()
    assert result.to_set() == {a, b}


def test_lub_abc_cross_container_different_args_for_variable():
    """lub(list[int], set[str]) merges args when for_variable=True."""
    import collections.abc
    from righttyper.generalize import lub

    a = TypeInfo.from_type(list).replace(args=(TypeInfo.from_type(int),))
    b = TypeInfo.from_type(set).replace(args=(TypeInfo.from_type(str),))

    result = lub(a, b, for_variable=True, accessed_attributes={"__iter__"})
    assert not result.is_union()
    assert isinstance(result.type_obj, type)
    assert issubclass(result.type_obj, collections.abc.Iterable)
    # Args should be merged: int|str
    assert result.args and _ti(result.args[0]).is_union()
    assert _ti(result.args[0]).to_set() == {TypeInfo.from_type(int), TypeInfo.from_type(str)}


# =============================================================================
# Tests inspired by mypy's JoinSuite (testtypes.py)
# =============================================================================

# Class hierarchy for mypy-style tests:
#     object
#    / |  \
#   A  D   F(abstract via ABC)
#  / \     |
# B   C    E

class A_mypy: pass
class B_mypy(A_mypy): pass
class C_mypy(A_mypy): pass
class D_mypy: pass

from abc import ABC
class F_mypy(ABC): pass
class E_mypy(F_mypy): pass


def test_lub_mypy_class_subtyping():
    """From mypy JoinSuite.test_class_subtyping."""
    from righttyper.generalize import lub
    a, b, c, d, o = (TypeInfo.from_type(t) for t in (A_mypy, B_mypy, C_mypy, D_mypy, object))
    # join(a, o) = o
    assert lub(a, o) == o
    # join(b, c) = a
    assert lub(b, c).type_obj is A_mypy
    # join(b, d) = union (only object in common, excluded)
    assert lub(b, d).is_union()
    assert lub(b, d).to_set() == {b, d}
    # join(a, d) = union
    assert lub(a, d).is_union()
    assert lub(a, d).to_set() == {a, d}


def test_lub_mypy_interface_types():
    """From mypy JoinSuite.test_join_interface_and_class_types.
    E implements F (abstract). join(e, f) = f."""
    from righttyper.generalize import lub
    e = TypeInfo.from_type(E_mypy)
    f = TypeInfo.from_type(F_mypy)
    assert lub(e, f) == f
    assert lub(f, e) == f


def test_lub_mypy_unrelated_with_object():
    """Unrelated types join to union (no useful base besides object)."""
    from righttyper.generalize import lub
    a = TypeInfo.from_type(A_mypy)
    d = TypeInfo.from_type(D_mypy)
    result = lub(a, d)
    assert result.is_union()
    assert result.to_set() == {a, d}


def test_lub_mypy_varlen_tuples():
    """From mypy JoinSuite.test_var_tuples.
    join(tuple[a], tuple[a, ...]) = tuple[a, ...]."""
    from righttyper.generalize import lub
    a_t = TypeInfo.from_type(A_mypy)
    fixed = TypeInfo.from_type(tuple).replace(args=(a_t,))
    varlen = TypeInfo.from_type(tuple).replace(args=(a_t, Ellipsis))
    assert lub(fixed, varlen) == varlen
    assert lub(varlen, fixed) == varlen
    # join(tuple[a, ...], tuple[()]) = tuple[a, ...]
    empty = TypeInfo.from_type(tuple).replace(args=((),))
    assert lub(varlen, empty) == varlen


def test_lub_callable_same_signature():
    """lub(Callable[[int], str], Callable[[int], str]) = Callable[[int], str]."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    params = TypeInfo.list([int_t])
    a = TypeInfo.from_type(collections.abc.Callable).replace(args=(params, str_t))
    assert lub(a, a) == a


def test_lub_callable_different_return():
    """lub(Callable[[int], int], Callable[[int], str]) merges return types (covariant)."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(collections.abc.Callable).replace(
        args=(TypeInfo.list([int_t]), int_t))
    b = TypeInfo.from_type(collections.abc.Callable).replace(
        args=(TypeInfo.list([int_t]), str_t))
    result = lub(a, b, for_variable=True)
    # Same params, different return → merge return to int|str
    assert result.type_obj is collections.abc.Callable
    assert not result.is_union()
    # Return type (last arg) should be a union
    assert _ti(result.args[-1]).is_union()
    assert _ti(result.args[-1]).to_set() == {int_t, str_t}
    # Params (first arg) should be unchanged
    assert _ti(result.args[0]).is_list()
    assert _ti(result.args[0]).args == (int_t,)


def test_lub_callable_different_params():
    """lub(Callable[[int], str], Callable[[float], str]) — different params stay as union."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    float_t = TypeInfo.from_type(float)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(collections.abc.Callable).replace(
        args=(TypeInfo.list([int_t]), str_t))
    b = TypeInfo.from_type(collections.abc.Callable).replace(
        args=(TypeInfo.list([float_t]), str_t))
    result = lub(a, b)
    # Different param types, invariant by default → union
    assert result.is_union()
    assert result.to_set() == {a, b}


def test_lub_callable_ellipsis_params():
    """lub(Callable[..., int], Callable[..., str]) merges return types."""
    import collections.abc
    from righttyper.generalize import lub
    int_t = TypeInfo.from_type(int)
    str_t = TypeInfo.from_type(str)
    a = TypeInfo.from_type(collections.abc.Callable).replace(args=(..., int_t))
    b = TypeInfo.from_type(collections.abc.Callable).replace(args=(..., str_t))
    result = lub(a, b, for_variable=True)
    assert result.type_obj is collections.abc.Callable
    assert not result.is_union()
    assert result.args[0] is ...
    assert _ti(result.args[-1]).is_union()
    assert _ti(result.args[-1]).to_set() == {int_t, str_t}


def test_lub_type_of_subtype():
    """lub(type[B], type[C]) = type[A] when B, C both extend A."""
    from righttyper.generalize import lub
    b_t = TypeInfo.from_type(type).replace(args=(TypeInfo.from_type(B_mypy),))
    c_t = TypeInfo.from_type(type).replace(args=(TypeInfo.from_type(C_mypy),))
    result = lub(b_t, c_t)
    assert result.type_obj is type
    assert not result.is_union()
    # type arg should be A (common base of B and C)
    assert _ti(result.args[0]).type_obj is A_mypy


def test_lub_type_of_same():
    """lub(type[B], type[B]) = type[B]."""
    from righttyper.generalize import lub
    b_t = TypeInfo.from_type(type).replace(args=(TypeInfo.from_type(B_mypy),))
    assert lub(b_t, b_t) == b_t


def test_lub_type_of_unrelated():
    """lub(type[A], type[D]) = type[A|D] (covariant merge of unrelated args)."""
    from righttyper.generalize import lub
    a_ti = TypeInfo.from_type(A_mypy)
    d_ti = TypeInfo.from_type(D_mypy)
    a_t = TypeInfo.from_type(type).replace(args=(a_ti,))
    d_t = TypeInfo.from_type(type).replace(args=(d_ti,))
    result = lub(a_t, d_t)
    assert result.type_obj is type
    assert not result.is_union()
    # Inner arg is the union A|D
    assert _ti(result.args[0]).is_union()
    assert _ti(result.args[0]).to_set() == {a_ti, d_ti}


# --- Rule 4b: bare generic subsumes parametrized ---

def test_lub_bare_type_subsumes_parametrized():
    """lub(type, type[int]) = type (bare type subsumes type[int])."""
    from righttyper.generalize import lub
    bare = TypeInfo.from_type(type)
    parametrized = TypeInfo.from_type(type).replace(args=(TypeInfo.from_type(int),))
    assert lub(bare, parametrized) == bare
    assert lub(parametrized, bare) == bare


def test_lub_bare_awaitable_subsumes_parametrized():
    """lub(Awaitable, Awaitable[int]) = Awaitable."""
    from righttyper.generalize import lub
    import collections.abc as abc
    bare = TypeInfo.from_type(abc.Awaitable)
    parametrized = TypeInfo.from_type(abc.Awaitable).replace(args=(TypeInfo.from_type(int),))
    assert lub(bare, parametrized) == bare
    assert lub(parametrized, bare) == bare


def test_lub_bare_async_iterable_subsumes_parametrized():
    """lub(AsyncIterable, AsyncIterable[str]) = AsyncIterable."""
    from righttyper.generalize import lub
    import collections.abc as abc
    bare = TypeInfo.from_type(abc.AsyncIterable)
    parametrized = TypeInfo.from_type(abc.AsyncIterable).replace(args=(TypeInfo.from_type(str),))
    assert lub(bare, parametrized) == bare
    assert lub(parametrized, bare) == bare


def test_lub_bare_reversible_subsumes_parametrized():
    """lub(Reversible, Reversible[float]) = Reversible."""
    from righttyper.generalize import lub
    import collections.abc as abc
    bare = TypeInfo.from_type(abc.Reversible)
    parametrized = TypeInfo.from_type(abc.Reversible).replace(args=(TypeInfo.from_type(float),))
    assert lub(bare, parametrized) == bare
    assert lub(parametrized, bare) == bare


def test_is_private_type():
    """Types in private modules without public re-export should be detected."""
    from righttyper.generalize import _is_private_type
    import io

    # _io.BytesIO: __module__='_io' but re-exported via io → not private
    assert not _is_private_type(io.BytesIO)
    assert not _is_private_type(io.StringIO)

    # builtins are not private
    assert not _is_private_type(int)
    assert not _is_private_type(dict)

    # A type genuinely stuck in a private module (no public re-export)
    private_type = type('HiddenType', (object,), {'__module__': '_secret_impl'})
    assert _is_private_type(private_type)

    # A _-prefixed class name in a public module
    private_name = type('_InternalHelper', (object,), {'__module__': 'mypkg.utils'})
    assert _is_private_type(private_name)


def test_lub_skips_private_mro_ancestors():
    """lub should not merge to a common ancestor defined in a private module."""
    from righttyper.generalize import lub

    # Build a class hierarchy where the common ancestor is in a private module
    Base = type('Base', (object,), {'__module__': '_internal.base', 'x': 1})
    A = type('A', (Base,), {'__module__': 'mypkg', 'x': 1})
    B = type('B', (Base,), {'__module__': 'mypkg', 'x': 1})

    a = TypeInfo.from_type(A)
    b = TypeInfo.from_type(B)
    result = lub(a, b)
    # Should NOT merge to Base (private) — should be a union
    assert result.is_union(), f"expected union, got {result}"


def test_merged_types_single_type_skips_private_ancestor():
    """merged_types with a single type and accessed_attributes should not
    generalize to an ancestor in a private module."""
    from righttyper.generalize import merged_types

    Base = type('Base', (object,), {'__module__': '_internal.base', 'do_thing': lambda self: None})
    Concrete = type('Concrete', (Base,), {'__module__': 'mypkg'})

    result = merged_types({TypeInfo.from_type(Concrete)}, accessed_attributes={'do_thing'})
    # Should stay as Concrete, not generalize to Base
    assert result == TypeInfo.from_type(Concrete)
