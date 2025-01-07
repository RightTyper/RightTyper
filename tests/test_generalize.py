from righttyper.righttyper_types import TypeInfo, TypeInfoSet
from righttyper.righttyper_utils import union_typeset_str
from typing import Sequence, cast


def generalize(
    samples: Sequence[tuple[TypeInfo, ...]],
    *,
    typevars: dict[tuple[TypeInfo, ...], str]|None = None
) -> list[str]|None:
    """
    Processes a sequence of samples observed for function parameters and return values, looking
    for patterns that can be replaced with type variables.  If no pattern is detected, the
    union of those types (per union_typeset_str) is built.

    samples: a sequence of tuples with type information. Each type in a tuple corresponds to
        a parameter (or return) type.
    typevars: a dictionary from type tuples (indicating a type usage pattern) to variable names.
    returns: a list of parameter (or return) type annotations.
    """

    # Ensure all samples are consistent (the same number of arguments)
    if any(len(t) != len(samples[0]) for t in samples[1:]):
        return None

    # By transposing the per-argument types, we obtain tuples with all the
    # various types seen for each argument.
    transposed = list(zip(*samples))

    def is_homogeneous_generic(types: tuple[TypeInfo, ...]) -> bool:
        """Whether the set only contains instances of a single, consistent generic type
           whose arguments are also all TypeInfo.
        """
        if not types:
            return False

        first = types[0]

        return all(
            t.module == first.module and 
            t.name == first.name and 
            len(t.args) == len(first.args) and
            all(isinstance(a, TypeInfo) for a in t.args)
            for t in types[1:]
        )

    from collections import Counter
    from typing import Iterator

    def expand_generics(types: tuple[TypeInfo, ...]) -> Iterator[tuple[TypeInfo, ...]]:
        yield types

        if is_homogeneous_generic(types):
            for i in range(len(types[0].args)):
                # cast dropping 'str' is checked by is_homogeneous_generic
                yield from expand_generics(cast(tuple[TypeInfo, ...], tuple(t.args[i] for t in types)))

    # Count the number of times a type usage pattern occurs, as we only want to generalize
    # if one occurs more than once (in more than one argument).
    occurrences: Counter[tuple[TypeInfo, ...]] = Counter()
    for types in transposed:
        occurrences.update([s for s in expand_generics(types)])

    if typevars is None:
        typevars = {}

    # Rebuild the argument list, defining and replacing type patterns with a type variable.
    def rebuild(types: tuple[TypeInfo, ...]) -> TypeInfo|str:
        if is_homogeneous_generic(types):
            args = tuple(
                rebuild(cast(tuple[TypeInfo, ...], tuple(t.args[i] for t in types)))
                for i in range(len(types[0].args))
            )

            return types[0].replace(args=args)

        if occurrences[types] > 1:
            if types not in typevars:
                typevars[types] = f"T{len(typevars)+1}"
            return typevars[types]

        return union_typeset_str(TypeInfoSet(types))

    return [str(rebuild(types)) for types in transposed]


def ti(name: str, **kwargs) -> TypeInfo:
    return TypeInfo(module='', name=name, **kwargs)


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
    typevars: dict[tuple[TypeInfo, ...], str] = {}
    assert generalize(samples, typevars=typevars) == ['T1', 'T1', 'T1']

    var = next(iter(typevars.keys()))
    assert union_typeset_str(TypeInfoSet(var)) == "bool|float|int"

    assert typevars[var] == 'T1'


def test_uses_existing_typevars():
    samples = [
        (ti('int'), ti('int'), ti('int')),
        (ti('bool'), ti('bool'), ti('bool')),
        (ti('float'), ti('float'), ti('float'))
    ]

    typevars: dict[tuple[TypeInfo, ...], str] = {}
    typevars[(ti('int'), ti('bool'), ti('float'))] = 'X'
    assert generalize(samples, typevars=typevars) == ['X', 'X', 'X']


def test_uniform_single_type_with_generic():
    samples = [
        (ti('int'), ti('int')),
        (ti('bool'), ti('bool')),
        (ti('X', args=(ti('foo'),)), ti('X', args=(ti('foo'),))),
    ]

    typevars: dict[tuple[TypeInfo, ...], str] = {}
    assert generalize(samples, typevars=typevars) == ['T1', 'T1']

    var = next(iter(typevars.keys()))
    assert union_typeset_str(TypeInfoSet(var)) == "X[foo]|bool|int"


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
        (ti('bool'), ti('None')),
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
        (ti('int'), ti('X', args=(ti('int'), "\"foo\""))),
        (ti('bool'), ti('X', args=(ti('bool'), "\"bar\""))),
    ]
    assert generalize(samples) == ['bool|int', 'X[bool, \"bar\"]|X[int, \"foo\"]']

    samples = [
        (ti('X', args=(ti('int'), "\"foo\"")),),
        (ti('X', args=(ti('bool'), "\"bar\"")),),
    ]
    assert generalize(samples) == ['X[bool, \"bar\"]|X[int, \"foo\"]']
