from typing import cast, Sequence, Iterator
import collections.abc as abc
from collections import defaultdict, Counter
from types import EllipsisType
from righttyper.typeinfo import TypeInfo, CallTrace
from righttyper.type_id import get_type_name
from righttyper.righttyper_types import cast_not_None
from righttyper.options import output_options


def merged_types(typeinfoset: set[TypeInfo]) -> TypeInfo:
    """Attempts to merge types in a set before forming their union."""

    if len(typeinfoset) > 1 and output_options.simplify_types:
        typeinfoset = simplify(typeinfoset)

    return TypeInfo.from_set(typeinfoset)


def simplify(typeinfoset: set[TypeInfo]) -> set[TypeInfo]:
    """Simplifies the set by replacing types with supertypes that contains
       all common attributes.
    """
    # Types we support simplifying
    simplifiable_types = set(
        t
        for t in typeinfoset
        if len(t.args) == 0                          # we don't compare arguments yet
        if not hasattr(t.type_obj, "__orig_class__") # we don't support generics yet
    )

    if not simplifiable_types:
        return typeinfoset

    other_types = typeinfoset - simplifiable_types 

    # generics whose incomplete (argumentless) forms default to Any arguments
    incomplete_types = set(
        t
        for t in simplifiable_types
        if (
            # TODO many more could be added here -- maybe every ABC?
            t.type_obj in (abc.Callable, abc.Iterator, abc.AsyncIterator, abc.Iterable,
                           abc.Generator, abc.AsyncGenerator, abc.Coroutine)
            or (type(t.type_obj) is type and issubclass(t.type_obj, abc.Container))
        )
    )

    if incomplete_types:
        # the incomplete form subsumes those with arguments: delete them
        other_types = set(
            t
            for t in other_types
            if not any(bc.type_obj is t.type_obj for bc in incomplete_types)
        )

    # Types we support merging
    mergeable_types = set(
        t
        for t in simplifiable_types
        if type(t.type_obj) is type     # we need a type_obj with __mro__ for merging
    )

    other_types |= (simplifiable_types - mergeable_types)

    if not mergeable_types:
        return other_types

    # TODO do we want to merge by protocol?  search for protocols in collections.abc types?

    def insert_numerics(mro: tuple[type, ...]) -> tuple[type, ...]:
        """Inserts numerics where applicable into an mro list
        
        This also preserves topological order.

        For example, suppose our type hierarchy looks like
        ```
             float int
                |   |
                A   B
        ```
        And we are given [A, float] for A and [B, int] for B

        According to [PEP 3141](https://peps.python.org/pep-3141/), this type hierarchy is equivalent to
        ```
             complex
                |
              float
                | \\
                |  int
                |   |
                A   B
        ```

        This method returns a new mro that is consistent with this type hierarchy
        ([A, float, complex] for A and [B, int, float, complex] for B)
        """
        numeric_hierarchy = frozenset({int, float, complex, object})
        numerics = [mro_type for mro_type in mro if mro_type in numeric_hierarchy]
        if len(numerics) <= 1:
            return mro
        new_mro = [mro_type for mro_type in mro if mro_type not in numeric_hierarchy]
        tower_index = min([int, float, complex, object].index(mro_type) for mro_type in numerics)
        new_mro.extend([int, float, complex, object][tower_index:])
        return tuple(new_mro)

    # FIXME besides attribute presence, we should check their types/signatures
    # FIXME we should check object attributes, not their classes'
    common_attributes = set.intersection(
        *(
            set(
                attr
                for attr in dir(cast_not_None(t.type_obj))
                if getattr(t.type_obj, attr, None) is not None
                if not attr.startswith("_") or attr.startswith("__")
            ) for t in mergeable_types
        )
    )

    # Get the superclasses, if any, that have all the common attributes
    common_supertypes = defaultdict(list)
    for t in mergeable_types:
        for base in insert_numerics(cast(type, t.type_obj).__mro__):
            if common_attributes.issubset(set(dir(base))):
                common_supertypes[base].append(t)

    # Unless "object" is in the set, it's likely too general to be useful
    # TODO why aren't the attributes enough to exclude "object" ?
    if object in common_supertypes and not any(t.type_obj is object for t in mergeable_types):
        del common_supertypes[object]

#    print("common_supertypes=", {k: [str(t) for t in v] for k, v in common_supertypes.items() if len(v)>1})

    # Since we want types that are as specific as possible, save the replacements in a separate set,
    # so that they don't get replaced as well.
    replacements = set()

    # 'defaultdict' retains insertion order, and 'sorted' is stable, so the first supertypes
    # on the list, after sorting by how many types they replace, are as specific as possible.
    for st, types in sorted(common_supertypes.items(), key=lambda item: -len(item[1])):
        if len(types) == 1:
            break   # we're not interested in 1-for-1 exchanges

        if any(t in mergeable_types for t in types):
            mergeable_types -= set(types)
            replacements.add(get_type_name(st))

    return mergeable_types | replacements | other_types


def generalize_jaxtyping(samples: Sequence[CallTrace]) -> Sequence[CallTrace]:
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
            isinstance(t.args[1], str)
        )

    def get_dims(t: TypeInfo) -> Sequence[str]:
        # str type already checked by is_jaxtyping_array
        return cast(str, t.args[1]).split()  # space separated dimensions

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
    names: dict[tuple[int, ...], tuple[str, ...]] = {}
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
                        f"{' '.join(dims)}"
                    )
                )
                for s, dims in zip(samples, tdims)
            ])
        else:
            results.append([s[argno] for s in samples])

    # Transpose once more to finish up
    return list(tuple(t) for t in zip(*results))


def generalize(samples: Sequence[CallTrace]) -> list[TypeInfo]|None:
    """
    Processes a sequence of samples observed for function parameters and return values, looking
    for patterns that can be replaced with type variables or, if does not detect a pattern,
    building type unions.

    samples: a sequence of tuples with type information. Each type in a tuple corresponds to
        a parameter (or return) type.
    returns: a list of parameter (or return) type annotations.
    """

    # Ensure all samples are consistent (the same number of arguments)
    if any(len(t) != len(samples[0]) for t in samples[1:]):
        return None

    samples = generalize_jaxtyping(samples)

    # By transposing the per-argument types, we obtain tuples with all the
    # various types seen for each argument.
    transposed = list(zip(*samples))

    def is_homogeneous(types: tuple[TypeInfo, ...]) -> bool:
        """Whether the tuple only contains instances of a single, consistent generic type
           whose arguments are all either TypeInfo or ellipsis.
        """
        if not types:
            return False

        first = types[0]

        return (
            all(
                isinstance(t, TypeInfo) and
                all(isinstance(a, (TypeInfo, EllipsisType)) for a in t.args)
                for t in types
            )
            and all(
                t.module == first.module and
                t.name == first.name and
                len(t.args) == len(first.args) and
                all((a is Ellipsis) == (first.args[i] is Ellipsis) for i, a in enumerate(t.args))
                for t in types[1:]
            )
        )

    # Count the number of times a type usage pattern occurs, as we only want to generalize
    # if one occurs more than once (in more than one argument).
    def expand_types(types: tuple[TypeInfo, ...]) -> Iterator[tuple[TypeInfo, ...]]:
        """Given a tuple of types used in an argument or return value, extracts the
           various type patterns enclosed in those type's arguments.
        """
        yield types

        if is_homogeneous(types):
            for i in range(len(types[0].args)):
                if types[0].args[i] is not Ellipsis:
                    yield from expand_types(cast(tuple[TypeInfo, ...], tuple(t.args[i] for t in types)))

    occurrences: Counter[tuple[TypeInfo, ...]] = Counter()
    for types in transposed:
        occurrences.update([s for s in expand_types(types)])

    # Rebuild the argument list, defining and replacing type patterns with a type variable.
    typevars: dict[tuple[TypeInfo, ...], TypeInfo] = {}

    def rebuild(types: tuple[TypeInfo, ...]) -> TypeInfo:
        # if the types look compatible, try to replace them with a single one using variable(s)
        if is_homogeneous(types):
            args = tuple(
                rebuild(tuple(cast(TypeInfo, t.args[i]) for t in types))
                if types[0].args[i] is not Ellipsis else Ellipsis
                for i in range(len(types[0].args))
            )

            return types[0].replace(args=args)

        combined = TypeInfo.from_set(set(types))

        # replace type sequence with a variable
        if occurrences[types] > 1 and combined.is_union():
            if types not in typevars:
                typevars[types] = combined.replace(typevar_index = len(typevars)+1)
            return typevars[types]

        return merged_types(combined.to_set())

    return [rebuild(types) for types in transposed]
