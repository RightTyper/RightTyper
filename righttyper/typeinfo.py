import itertools
from typing import Sequence, Iterator, cast
from .righttyper_types import TypeInfo, NoneTypeInfo, TypeInfoSet, Typename, TYPE_OBJ_TYPES
from .righttyper_utils import get_main_module_fqn


def union_typeset(typeinfoset: TypeInfoSet) -> TypeInfo:
    if not typeinfoset:
        return TypeInfo.from_type(type(None)) # Never observed any types.

    if len(typeinfoset) == 1:
        return next(iter(typeinfoset))

    if super := find_most_specific_common_superclass_by_name(typeinfoset):
        return super

    # merge similar generics
    if any(t.args for t in typeinfoset):
        typeinfoset = TypeInfoSet({*typeinfoset})   # avoid modifying

        def group_key(t):
            return t.module, t.name, all(isinstance(arg, TypeInfo) for arg in t.args), len(t.args)
        group: Iterator[TypeInfo]|TypeInfoSet
        for (mod, name, all_info, nargs), group in itertools.groupby(
            sorted(typeinfoset, key=group_key),
            group_key
        ):
            if all_info:
                group = set(group)
                first = next(iter(group))
                typeinfoset -= group
                typeinfoset.add(first.replace(args=tuple(
                        union_typeset(TypeInfoSet({
                            cast(TypeInfo, member.args[i]) for member in group
                        }))
                        for i in range(nargs)
                    )
                ))

    # TODO merge jaxtyping annotations by shape

    return TypeInfo.from_set(typeinfoset)


def union_typeset_str(typeinfoset: TypeInfoSet) -> Typename:
    return Typename(str(union_typeset(typeinfoset)))


def find_most_specific_common_superclass_by_name(typeinfoset: TypeInfoSet) -> TypeInfo|None:
    if any(t.type_obj is None for t in typeinfoset):
        return None

    common_superclasses = set.intersection(
        *(set(cast(TYPE_OBJ_TYPES, t.type_obj).__mro__) for t in typeinfoset)
    )

    common_superclasses.discard(object) # not specific enough to be useful

    if not common_superclasses:
        return None

    specific = max(
            common_superclasses,
            key=lambda cls: cls.__mro__.index(object),
    )

    module = specific.__module__ if specific.__module__ != '__main__' else get_main_module_fqn()
    return TypeInfo(module, specific.__qualname__, type_obj=specific)



def generalize(samples: Sequence[tuple[TypeInfo, ...]]) -> list[TypeInfo]|None:
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

    typevars: dict[tuple[TypeInfo, ...], TypeInfo] = {}

    # Rebuild the argument list, defining and replacing type patterns with a type variable.
    def rebuild(types: tuple[TypeInfo, ...]) -> TypeInfo:
        if is_homogeneous_generic(types):
            args = tuple(
                rebuild(cast(tuple[TypeInfo, ...], tuple(t.args[i] for t in types)))
                for i in range(len(types[0].args))
            )

            return types[0].replace(args=args)

        if occurrences[types] > 1:
            if types not in typevars:
                typevars[types] = TypeInfo.from_set(
                    TypeInfoSet(types),
                    typevar_index = len(typevars)+1
                )
            return typevars[types]

        return union_typeset(TypeInfoSet(types))

    return [rebuild(types) for types in transposed]
