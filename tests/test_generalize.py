from righttyper.righttyper_types import TypeInfo, TypeInfoSet
from righttyper.righttyper_utils import union_typeset_str
import pytest
from typing import Iterator

def ti(name: str, **kwargs) -> TypeInfo:
    return TypeInfo(module='', name=name, **kwargs)

def get_source(ti: TypeInfo) -> str:
    f"{ti.module}.{ti.name}" if ti.module else ti.name

NoneTypeInfo = ti('None')

def generalize(samples: list[tuple[TypeInfo, ...]]) -> list[str]:
    # ensure all samples are consistent (the same number of arguments)
    if any (len(t) != len(samples[0]) for t in samples[1:]):
        return None

    typevars: dict[tuple[TypeInfo, ...], str] = {}
    parameters = []

    transposed = list(zip(*samples))

    from collections import Counter

    def expand_generics(types: tuple[TypeInfo, ...]) -> Iterator[tuple[TypeInfo, ...]]:
        print(f"types={[str(t) for t in types]}")
        yield types

        first = types[0]
        if first.args and all(
            t.module == first.module and 
            t.name == first.name and 
            len(t.args) == len(first.args)
            for t in types[1:]
        ):
            for i in range(len(first.args)):
                yield from expand_generics(tuple(t.args[i] for t in types))

    occurrences = Counter()
    for types in transposed:
        occurrences.update([s for s in expand_generics(types)])

    def rebuild(types: tuple[TypeInfo, ...]) -> TypeInfo|str:
        first = types[0]
        if first.args and all(
            t.module == first.module and 
            t.name == first.name and 
            len(t.args) == len(first.args)
            for t in types[1:]
        ):
            args = []
            for i in range(len(first.args)):
                args.append(rebuild(tuple(t.args[i] for t in types)))

            # FIXME replace with TypeInfo.with_changes()
            return TypeInfo(module=first.module, name=first.name, args=tuple(args))

        if occurrences[types] > 1:
            if types not in typevars:
                typevars[types] = f"T{len(typevars)+1}"
            return typevars[types]

        return union_typeset_str(TypeInfoSet(types))

    for types in transposed:
        parameters.append(str(rebuild(types)))
    
    return parameters


def generalize2(tuples: list[tuple[type, ...]]) -> list[str]:
    tilist = []
    for tupl in tuples:
        tilist.append(tuple(
            NoneTypeInfo if t is None else ti(t.__name__)
            for t in tupl
        ))

    return generalize(tilist)


def test_no_tuples():
    assert generalize2([]) == []


def test_single_tuple():
    assert generalize2([(int, float, str)]) == ['int', 'float', 'str']


def test_none():
    assert generalize2([(int, None)]) == ['int', 'None']

    assert generalize2([
        (int, None),
        (int, bool)
    ]) == ['int', 'bool|None']


def test_uniform_single_type():
    samples = [
        (int, int, int),
        (bool, bool, bool),
        (float, float, float)
    ]
    assert generalize2(samples) == ['T1', 'T1', 'T1']

def test_uniform_single_type_with_generic():
    samples = [
        (ti('int'), ti('int')),
        (ti('bool'), ti('bool')),
        (ti('X', args=(ti('foo'),)), ti('X', args=(ti('foo'),))),
    ]
    assert generalize(samples) == ['T1', 'T1']

def test_first_same_then_different():
    samples = [
        (int, int),
        (bool, bool),
        (int, bool)
    ]
    assert generalize2(samples) == ['bool|int', 'bool|int']


def test_mixed_with_constant_types():
    samples = [
        (int, str, int),
        (bool, str, float),
        (float, str, bool)
    ]
    assert generalize2(samples) == ['bool|float|int', 'str', 'bool|float|int']


def test_shared_variability():
    samples = [
        (int, int, bool, int),
        (float, float, bool, float)
    ]
    assert generalize2(samples) == ['T1', 'T1', 'bool', 'T1']


def test_multiple_length_tuples():
    samples = [
        (int, int),
        (int, int, int),
    ]
    assert generalize2(samples) is None


def test_all_distinct_types():
    samples = [
        (int, str, float, bool),
        (float, str, bool, int)
    ]
    assert generalize2(samples) == ['float|int', 'str', 'bool|float', 'bool|int']


def test_generic():
    samples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
    ]
    assert generalize(samples) == ['T1', 'list[T1]']


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
