from righttyper.righttyper_types import TypeInfo
import pytest

def ti(name: str, **kwargs) -> TypeInfo:
    return TypeInfo(module='', name=name, **kwargs)

NoneTypeInfo = ti('None')

def generalize(samples: list[tuple[TypeInfo, ...]]) -> list[str]:
    # ensure all samples are consistent (the same number of arguments)
    if any (len(t) != len(samples[0]) for t in samples[1:]):
        return None

    typevars: dict[tuple[TypeInfo, ...], str] = {}
    parameters = []

    transposed = list(zip(*samples))

    for i, types in enumerate(transposed):
        if types in typevars:
            parameters.append(typevars[types])
        else:
            if types in transposed[i+1:]:
                typevars[types] = f"T{len(typevars)+1}"
                parameters.append(typevars[types])
            else:
                types = tuple(dict.fromkeys(types))
                if NoneTypeInfo in types:
                    types = (*tuple(t for t in types if t != NoneTypeInfo), None)
                parameters.append('|'.join([str(t) for t in types]))
    
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
    tuples = [
        (int, int, int),
        (bool, bool, bool),
        (float, float, float)
    ]
    assert generalize2(tuples) == ['T1', 'T1', 'T1']

def test_first_same_then_different():
    tuples = [
        (int, int),
        (bool, bool),
        (int, bool)
    ]
    assert generalize2(tuples) == ['int|bool', 'int|bool']

def test_mixed_with_constant_types():
    tuples = [
        (int, str, int),
        (bool, str, float),
        (float, str, bool)
    ]
    assert generalize2(tuples) == ['int|bool|float', 'str', 'int|float|bool']

def test_shared_variability():
    tuples = [
        (int, int, bool, int),
        (float, float, bool, float)
    ]
    assert generalize2(tuples) == ['T1', 'T1', 'bool', 'T1']

def test_multiple_length_tuples():
    tuples = [
        (int, int),
        (int, int, int),
    ]
    assert generalize2(tuples) is None

def test_all_distinct_types():
    tuples = [
        (int, str, float, bool),
        (float, str, bool, int)
    ]
    assert generalize2(tuples) == ['int|float', 'str', 'float|bool', 'bool|int']

@pytest.mark.xfail()
def test_generic():
    tuples = [
        (ti('int'), ti('list', args=(ti('int'),))),
        (ti('float'), ti('list', args=(ti('float'),))),
    ]
    assert generalize(tuples) == ['T1', 'list[T1]']
