import itertools
from typing import Sequence, Iterator, cast
from .righttyper_types import TypeInfo, TYPE_OBJ_TYPES, NoneTypeInfo
from .righttyper_utils import get_main_module_fqn
from collections import Counter


# TODO integrate these into TypeInfo?

class SimplifyGeneratorsTransformer(TypeInfo.Transformer):
    def visit(self, node: TypeInfo) -> TypeInfo:
        if (
            node.module == "typing"
            and node.name == "Generator"
            and len(node.args) == 3
            and node.args[1] == NoneTypeInfo
            and node.args[2] == NoneTypeInfo
        ):
            return TypeInfo("typing", "Iterator", (node.args[0],))
        
        return super().visit(node)


def merged_types(typeinfoset: set[TypeInfo]) -> TypeInfo:
    """Attempts to merge types in a set before forming their union."""

    if len(typeinfoset) > 1:
        if sclass := find_most_specific_common_superclass(typeinfoset):
            return sclass

        # merge similar generics
        if any(t.args for t in typeinfoset):
            typeinfoset = set(typeinfoset)   # avoid modifying argument

            # TODO group by superclass/protocol when possible, so that these can be merged
            # e.g.: list[int], Sequence[int]

            def group_key(t):
                return t.module, t.name, all(isinstance(arg, TypeInfo) for arg in t.args), len(t.args)
            group: Iterator[TypeInfo]|set[TypeInfo]
            for (mod, name, all_info, nargs), group in itertools.groupby(
                sorted(typeinfoset, key=group_key),
                group_key
            ):
                if all_info:
                    group = set(group)
                    first = next(iter(group))
                    typeinfoset -= group
                    typeinfoset.add(first.replace(args=tuple(
                            merged_types({
                                cast(TypeInfo, member.args[i]) for member in group
                            })
                            for i in range(nargs)
                        )
                    ))

    tr = SimplifyGeneratorsTransformer()
    typeinfoset = set(
        tr.visit(it)
        for it in typeinfoset
    )
    
    return TypeInfo.from_set(typeinfoset)


def find_most_specific_common_superclass(typeinfoset: set[TypeInfo]) -> TypeInfo|None:
    if any(t.type_obj is None for t in typeinfoset):    # we require type_obj for this
        return None

    # TODO do we want to merge by protocol?  search for protocols in collections.abc types?
    # TODO we could also merge on portions of the set

    # FIXME besides attribute presence, we should check their types/signatures
    # FIXME we should check object attributes, not their classes'
    common_attributes = set.intersection(
        *(set(dir(cast(TYPE_OBJ_TYPES, t.type_obj))) for t in typeinfoset)
    )

    # Get the superclasses, if any, that have all the common attributes
    common_superclasses = set.intersection(
        *(
            set(
                base
                for base in cast(TYPE_OBJ_TYPES, t.type_obj).__mro__
                if common_attributes.issubset(set(dir(base)))
            )
            for t in typeinfoset
        )
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


def generalize_jaxtyping(samples: Sequence[tuple[TypeInfo, ...]]) -> Sequence[tuple[TypeInfo, ...]]:
    # Ensure all samples are consistent (the same number of arguments)
    if any(len(t) != len(samples[0]) for t in samples[1:]):
        return samples

    # With a single sample we don't try to infer dimension variables:
    # any matches could easily be coincidence.
    if len(samples) < 2:
        return samples

    # Transpose to get parameters together
    transposed = list(zip(*samples))

    def is_jaxtyping_array(t: TypeInfo) -> bool:
        return (
            t.module == 'jaxtyping' and
            len(t.args) == 2 and
            isinstance(t.args[1], str) and
            t.args[1][0] in ('"', "'") and t.args[1][-1] == t.args[1][0]
        )

    def get_dims(t: TypeInfo) -> Sequence[str]:
        # str type already checked by is_jaxtyping_array
        return cast(str, t.args[1])[1:-1].split()  # space separated dimensions within quotes

    # Get the set of dimensions seen for each consistent jaxtyping array
    dimensions = {
        argno: list(zip(*(get_dims(t) for t in arg)))
        for argno, arg in enumerate(transposed)
        if all(is_jaxtyping_array(t) for t in arg)
        if len(set(len(get_dims(t)) for t in arg)) == 1 # consistent no. of dimensions
    }

    if not dimensions:
        return samples

    occurrences = Counter(dims for argdims in dimensions.values() for dims in argdims)

    # Assign names to common dimensions
    names: dict[tuple, tuple[str, ...]] = {}
    for argdims in dimensions.values():
        for i, dims in enumerate(argdims):
            if dims in names:
                argdims[i] = names[dims]
            elif occurrences[dims] > 1:
                argdims[i] = names[dims] = (f"D{len(names)+1}",) * len(dims)

    # Replace args where needed
    results = []
    for argno in range(len(samples[0])):
        if argno in dimensions:
            tdims = list(zip(*dimensions[argno]))
            results.append([
                s[argno].replace(args=(
                        s[argno].args[0],
                        f"\"{' '.join(dims)}\""
                    )
                )
                for s, dims in zip(samples, tdims)
            ])
        else:
            results.append([s[argno] for s in samples])

    # Transpose once more to finish up
    return list(tuple(t) for t in zip(*results))


def generalize(samples: Sequence[tuple[TypeInfo, ...]]) -> list[TypeInfo]|None:
    """
    Processes a sequence of samples observed for function parameters and return values, looking
    for patterns that can be replaced with type variables or, if does not detect a pattern,
    building type unions.

    samples: a sequence of tuples with type information. Each type in a tuple corresponds to
        a parameter (or return) type.
    typevars: a dictionary from type tuples (indicating a type usage pattern) to variable names.
    returns: a list of parameter (or return) type annotations.
    """

    # Ensure all samples are consistent (the same number of arguments)
    if any(len(t) != len(samples[0]) for t in samples[1:]):
        return None

    samples = generalize_jaxtyping(samples)

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

        return (
            all(
                all(isinstance(a, TypeInfo) for a in t.args)
                for t in types
            )
            and all(
                t.module == first.module and
                t.name == first.name and
                len(t.args) == len(first.args) and
                all(isinstance(a, TypeInfo) for a in t.args)
                for t in types[1:]
            )
        )

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

            return SimplifyGeneratorsTransformer().visit(types[0].replace(args=args))

        if occurrences[types] > 1:
            if types not in typevars:
                typevars[types] = TypeInfo.from_set(
                    set(types),
                    typevar_index = len(typevars)+1
                )
            return typevars[types]

        return merged_types(set(types))


    return [rebuild(types) for types in transposed]
