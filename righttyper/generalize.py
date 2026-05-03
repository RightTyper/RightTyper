from typing import cast, Sequence, Iterator, Any, Never, NoReturn
from abc import ABCMeta
import collections.abc as abc
from collections import Counter
from functools import cache
from types import EllipsisType
import sys
from righttyper.typeinfo import TypeInfo, ListTypeInfo, CallTrace
from righttyper.type_id import get_type_name
from righttyper.options import output_options


# Types that are covariant (immutable), so merging their type arguments
# is safe even for function parameters and return types.
_COVARIANT_TYPES = (tuple, frozenset)

# Generic types where bare form (no args) means "Any args" and subsumes parametrized forms.
# Containers are handled separately via issubclass(t, abc.Container).
# abc.Callable etc. are typing special forms, not types — silence mypy.
_BARE_SUBSUMES: set[type] = cast(set[type], {
    abc.Callable, abc.Iterator, abc.AsyncIterator, abc.Iterable,
    abc.Generator, abc.AsyncGenerator, abc.Coroutine,
    abc.Awaitable, abc.AsyncIterable, abc.Reversible, abc.MappingView, type,
})

# All generic container ABCs from collections.abc, ordered most-specific-first.
_CONTAINER_ABCS: list[type] = [
    obj for obj in reversed(list(vars(abc).values()))
    if isinstance(obj, type) and hasattr(obj, '__class_getitem__') and issubclass(obj, abc.Container)
]



def is_homogeneous(types: tuple[TypeInfo, ...] | list[TypeInfo]) -> bool:
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
            t.code_id == first.code_id and
            len(t.args) == len(first.args) and
            all((a is Ellipsis) == (first.args[i] is Ellipsis) for i, a in enumerate(t.args)) and
            # ListTypeInfo args (e.g. Callable param lists) must have the same arity
            all(
                len(cast(TypeInfo, t.args[i]).args) == len(cast(TypeInfo, first.args[i]).args)
                for i in range(len(first.args))
                if isinstance(first.args[i], ListTypeInfo)
            )
            for t in list(types)[1:]
        )
    )


# Precomputed ABC data for structural matching in lub.
def _build_abc_own_attrs() -> dict[type, frozenset[str]]:
    """Precompute the own attributes (excluding object) for each collections.abc ABC."""
    result: dict[type, frozenset[str]] = {}
    for obj in vars(abc).values():
        if isinstance(obj, ABCMeta):
            attrs: set[str] = set()
            for cls in obj.__mro__:
                if cls is object:
                    break
                attrs |= set(cls.__dict__)
            result[obj] = frozenset(attrs)
    return result

_abc_own_attrs = _build_abc_own_attrs()


def _find_common_container_abc(t1: TypeInfo, t2: TypeInfo) -> type | None:
    """Find the most specific generic container ABC both types implement.

    Caller must ensure both type_objs are real types (not None or special forms).
    """
    # Caller's contract guarantees both type_objs are real types.
    o1, o2 = cast(type, t1.type_obj), cast(type, t2.type_obj)
    for g in _CONTAINER_ABCS:
        if issubclass(o1, g) and issubclass(o2, g):
            return g
    return None


@cache
def _is_private_type(cls: type) -> bool:
    """Check if cls is defined in a private module without public re-export.

    This partially duplicates TypeMap's is_private logic; done here to avoid
    threading the TypeMap through to lub.
    """
    # A _-prefixed class name signals a private implementation detail,
    # regardless of which module it lives in.
    name = getattr(cls, '__name__', '') or ''
    if name.startswith('_') and not name.startswith('__'):
        return True
    mod = getattr(cls, '__module__', '') or ''
    if mod == '__main__' or not any(p.startswith('_') for p in mod.split('.')):
        return False
    # Private module — but check if any public module re-exports this type
    name = getattr(cls, '__name__', '')
    for m in sys.modules.values():
        m_name = getattr(m, '__name__', '')
        if m_name and not any(p.startswith('_') for p in m_name.split('.')):
            if getattr(m, name, None) is cls:
                return False
    return True


# Numeric tower: int <: float <: complex (PEP 3141)
_NUMERIC_TOWER = {int: float, float: complex}


def _is_subtype(a: type, b: type) -> bool:
    """Check if a is a subtype of b, including the numeric tower."""
    if a is b:
        return True
    if issubclass(a, b):
        return True
    # Numeric tower: int <: float <: complex. Walk a's MRO for the nearest
    # tower entry, then promote up the tower checking against b.
    for ancestor in a.__mro__:
        if ancestor in _NUMERIC_TOWER:
            t = ancestor
            while t in _NUMERIC_TOWER:
                t = _NUMERIC_TOWER[t]
                if issubclass(t, b):
                    return True
            break
    return False


def lub(
    a: TypeInfo,
    b: TypeInfo,
    for_variable: bool = False,
    accessed_attributes: set[str] | None = None,
) -> TypeInfo:
    """Compute the least upper bound (most specific common supertype) of two types.

    Returns a single TypeInfo that contains both a and b. Falls back to
    a union (a | b) when no better merge is available.
    """
    # Rule 1: Identity
    if a == b:
        return a

    # Rule 2: Never/NoReturn subsumption
    if a.type_obj in (Never, NoReturn):
        return b
    if b.type_obj in (Never, NoReturn):
        return a

    # Rule 3: Any subsumption
    if a.type_obj is Any:
        return a
    if b.type_obj is Any:
        return b

    # Reduce Generator/AsyncGenerator to Iterator/AsyncIterator before merging,
    # so their extra args (send, return) don't leak into ABC matches.
    def _reduce_generator(t: TypeInfo) -> TypeInfo:
        if t.type_obj is abc.Generator and len(t.args) == 3:
            return TypeInfo.from_type(abc.Iterator, args=(t.args[0],) if isinstance(t.args[0], TypeInfo) else ())
        if t.type_obj is abc.AsyncGenerator and len(t.args) == 2:
            return TypeInfo.from_type(abc.AsyncIterator, args=(t.args[0],) if isinstance(t.args[0], TypeInfo) else ())
        return t

    a, b = _reduce_generator(a), _reduce_generator(b)
    if a == b:
        return a

    # Without a real type (could be None or a typing special form),
    # no merging rules apply.
    if not isinstance(a.type_obj, type) or not isinstance(b.type_obj, type):
        return TypeInfo.from_set_new({a, b})

    # Rule 4: Subtype check (MRO + numeric tower)
    if not a.args and not b.args:
        if _is_subtype(a.type_obj, b.type_obj):
            return b
        if _is_subtype(b.type_obj, a.type_obj):
            return a

    # Rules 4b + 5: Same type — bare subsumption and arg merging.
    if a.type_obj is b.type_obj:
        # Rule 4b: Bare generic subsumes parametrized (list subsumes list[int]).
        if a.type_obj in _BARE_SUBSUMES or issubclass(a.type_obj, abc.Container):
            if not a.args and b.args:
                return a
            if a.args and not b.args:
                return b

        # Rule 5: Same container, different args
        if a.args and b.args:
            # Rule 5a: Varlen tuple subsumes fixed tuple (different arg lengths OK)
            if a.type_obj is tuple:
                def _is_varlen(t: TypeInfo) -> bool:
                    return len(t.args) == 2 and t.args[1] is Ellipsis and isinstance(t.args[0], TypeInfo)
                def _is_fixed(t: TypeInfo) -> bool:
                    return not _is_varlen(t) and all(isinstance(x, TypeInfo) for x in t.args)

                for vl, other in ((a, b), (b, a)):
                    if _is_varlen(vl) and (_is_fixed(other) or other.args == ((),)):
                        elem = cast(TypeInfo, vl.args[0])
                        if other.args == ((),) or all(type_contains(elem, cast(TypeInfo, x)) for x in other.args):
                            return vl

            # Rule 5b: Same container, empty subsumed by non-empty.
            # For tuples, skip — rule 5d will merge to varlen instead.
            if a.type_obj is not tuple:
                if _is_empty_container(a):
                    return b
                if _is_empty_container(b):
                    return a

            # Rule 5c: Same container, same arg count — merge args
            if len(a.args) == len(b.args):
                # type[X] is covariant (type[B] <: type[A] when B <: A)
                is_covariant = issubclass(a.type_obj, _COVARIANT_TYPES) or a.type_obj is type
                if for_variable or is_covariant:
                    can_merge = all(
                        (isinstance(aa, TypeInfo) and isinstance(ba, TypeInfo))
                        or aa is ba
                        for aa, ba in zip(a.args, b.args)
                    )
                    if can_merge:
                        merged_args = tuple(
                            lub(cast(TypeInfo, aa), cast(TypeInfo, ba), for_variable=True)
                            if isinstance(aa, TypeInfo) and isinstance(ba, TypeInfo)
                            else aa
                            for aa, ba in zip(a.args, b.args)
                        )
                        return a.replace(args=merged_args)

            # Rule 5d: Different-length or empty fixed tuples → varlen tuple
            # tuple[int] | tuple[int, str] → tuple[int|str, ...]
            # tuple[()] | tuple[int] → tuple[int, ...]
            if a.type_obj is tuple and (len(a.args) != len(b.args)
                                        or a.args == ((),) or b.args == ((),)):
                all_elems: set[TypeInfo] = {x for x in (*a.args, *b.args) if isinstance(x, TypeInfo)}
                if all_elems:
                    merged_elem = TypeInfo.from_set(all_elems)
                    return a.replace(args=(merged_elem, Ellipsis))
                # Both empty → stay as empty
                if a.args == ((),):
                    return a

    # Rule 6: Empty container + non-empty container → MRO common supertype
    # with the non-empty container's args (the empty side contributes nothing).
    # E.g., dict[Never,Never] + defaultdict[str, list[X]] → dict[str, list[X]].
    if a.type_obj is not b.type_obj:  # different containers only
        empty, nonempty = (a, b) if _is_empty_container(a) else (b, a) if _is_empty_container(b) else (None, None)
        if empty is not None and nonempty is not None and nonempty.args:
            e_obj, ne_obj = cast(type, empty.type_obj), cast(type, nonempty.type_obj)
            # MRO walk: find nearest common supertype (includes ABCs
            # explicitly inherited, e.g. Generator → Iterator → Iterable)
            common: type | None = None
            for cls in ne_obj.__mro__:
                if issubclass(e_obj, cls) and cls is not object:
                    common = cls
                    break
            else:
                # Fallback: virtual ABC registration (e.g. list + dict → Collection)
                if isinstance(nonempty.args[0], TypeInfo):
                    common = _find_common_container_abc(empty, nonempty)

            if common is not None:
                # Mappings (abc or concrete) take 2 type params; all other
                # containers take 1.
                nparams = 2 if issubclass(common, abc.Mapping) else 1
                args = tuple(a for a in nonempty.args[:nparams] if isinstance(a, TypeInfo))
                return get_type_name(common).replace(args=args)

    # Rule 7: MRO common supertype (non-generic types only).
    if not a.args and not b.args:
        if accessed_attributes:
            common_attrs = accessed_attributes
        else:
            # Without accessed_attributes, use dir() intersection as safety filter:
            # only merge to a supertype that has all the shared attributes.
            common_attrs = (
                {attr for attr in dir(a.type_obj)
                 if getattr(a.type_obj, attr, None) is not None
                 if not attr.startswith("_") or attr.startswith("__")}
                &
                {attr for attr in dir(b.type_obj)
                 if getattr(b.type_obj, attr, None) is not None
                 if not attr.startswith("_") or attr.startswith("__")}
            )
        a_mro = set(a.type_obj.__mro__)
        for base in b.type_obj.__mro__:
            if base in a_mro and base is not object:
                if base in (int, float, complex):
                    continue
                if _is_private_type(base):
                    continue
                if common_attrs.issubset(dir(base)):
                    return get_type_name(base)

    # Rule 8: ABC matching (when accessed_attributes available)
    if accessed_attributes:
        candidates = [g for g, attrs in _abc_own_attrs.items()
                      if accessed_attributes <= attrs
                      and issubclass(a.type_obj, g)
                      and issubclass(b.type_obj, g)]
        if candidates:
            best = max(candidates, key=lambda g: len(g.__mro__))
            # For generics: try to merge element types
            if a.args or b.args:
                a_ti = [x for x in a.args if isinstance(x, TypeInfo)] if a.args else []
                b_ti = [x for x in b.args if isinstance(x, TypeInfo)] if b.args else []
                n = min(len(a_ti), len(b_ti))
                if n > 0:
                    # If either type is immutable, the function can't mutate
                    # the container, so the merge is effectively covariant.
                    # FIXME: this assumes no type narrowing (isinstance checks);
                    # if the code narrows, the mutable branch could be written to.
                    either_immutable = (
                        issubclass(a.type_obj, _COVARIANT_TYPES)
                        or issubclass(b.type_obj, _COVARIANT_TYPES)
                        or a.type_obj is type or b.type_obj is type
                    )
                    if for_variable or either_immutable:
                        merged_args = tuple(
                            lub(a_ti[i], b_ti[i], for_variable=True)
                            for i in range(n)
                        )
                        return get_type_name(best).replace(args=merged_args)
                    # Invariant: only merge if args are identical
                    if a_ti[:n] == b_ti[:n]:
                        return get_type_name(best).replace(args=tuple(a_ti[:n]))
                # Can't merge args — fall through to union
            else:
                return get_type_name(best)

    # Rule 9: Fallback — union
    return TypeInfo.from_set_new({a, b})


def _merge_set(
    typeinfoset: set[TypeInfo],
    for_variable: bool = False,
    accessed_attributes: set[str] | None = None,
) -> TypeInfo:
    """Reduce a set of types using pairwise lub, then form the final union."""
    if not typeinfoset:
        return TypeInfo.from_type(Never)

    # Single-type generalization: if accessed_attributes are provided,
    # walk up the MRO to find a more general base type. Each accessed attribute
    # must resolve to the same object on the base as on t — i.e. the base must
    # not be overridden in a closer descendant. Otherwise we'd simplify to a
    # base whose attribute (e.g. a method with a narrower signature) doesn't
    # match the actual usage at runtime.
    # De-privatize: if the sole observed type is private, walk up the MRO
    # to find the nearest public ancestor that still has all accessed
    # attributes.  Does NOT generalize public types.
    if len(typeinfoset) == 1:
        t = next(iter(typeinfoset))
        if isinstance(t.type_obj, type) and not t.args and _is_private_type(t.type_obj):
            sentinel = object()
            check_attrs = accessed_attributes or frozenset(
                attr for attr in dir(t.type_obj)
                if getattr(t.type_obj, attr, None) is not None
                if not attr.startswith("_") or attr.startswith("__")
            )
            for base in t.type_obj.__mro__:
                if base is t.type_obj or base is object:
                    continue
                if _is_private_type(base):
                    continue
                # First public ancestor: de-privatize if it has all
                # accessed attributes, otherwise keep the private name.
                if all(
                    (a := getattr(base, attr, sentinel)) is not sentinel
                    and a is getattr(t.type_obj, attr, sentinel)
                    for attr in check_attrs
                ):
                    return get_type_name(base)
                break

    # Flatten any non-typevar unions in the input so each leaf type
    # participates in the pairwise reduction. Otherwise lub treats `int|str`
    # as opaque (its type_obj isn't a real `type`) and never gets to compare
    # `bool` against `int` to discover `bool ⊆ int`. Typevar'd unions stay
    # opaque — they represent a single named variable, not a leaf set.
    types = list({
        leaf
        for t in typeinfoset
        for leaf in ({t} if t.is_typevar() else t.to_set())
    })

    # Iteratively reduce: try to merge pairs until stable
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(types):
            j = i + 1
            while j < len(types):
                merged = lub(types[i], types[j], for_variable, accessed_attributes)
                if not merged.is_union():
                    # lub produced a single type — replace both with it
                    types[i] = merged
                    types.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    # lub already handled simplification; just form the union
    return TypeInfo.from_set_new(set(types))


def type_contains(a: TypeInfo, b: TypeInfo) -> bool:
    """Check if a contains b (i.e., b's types are a subset of a's).

    This is recursive: for containers, all args of b must be contained by
    corresponding args of a.
    """
    if a == b:
        return True

    a_set = a.to_set()
    b_set = b.to_set()

    # Direct subset: b's types are all in a's types
    if b_set <= a_set:
        return True

    # Container containment: same generic, each arg of b contained by corresponding arg of a
    if (a.module == b.module and
        a.name == b.name and
        len(a.args) == len(b.args) and
        a.args and
        all(isinstance(arg, TypeInfo) for arg in a.args) and
        all(isinstance(arg, TypeInfo) for arg in b.args)):
        return all(
            type_contains(cast(TypeInfo, aa), cast(TypeInfo, ba))
            for aa, ba in zip(a.args, b.args)
        )

    return False


def _is_empty_container(t: TypeInfo) -> bool:
    """Check if t represents an empty container (e.g., tuple[()], list[Never], dict[Never, Never])."""
    if t.type_obj is tuple and t.args == ((),):
        return True
    return bool(t.args
                and isinstance(t.type_obj, type) and issubclass(t.type_obj, abc.Container)
                and all(isinstance(a, TypeInfo) and a.type_obj is Never
                        for a in t.args))


def merged_types(
    typeinfoset: set[TypeInfo],
    for_variable: bool = False,
    accessed_attributes: set[str] | None = None,
) -> TypeInfo:
    """Attempts to merge types in a set before forming their union."""
    if output_options.simplify_types:
        # Static-analysis simplification (single-type MRO walk + Rule-8 ABC matching)
        # is gated separately; honor the flag here so callers don't need to.
        if not output_options.use_attribute_simplification:
            accessed_attributes = None
        return _merge_set(typeinfoset, for_variable, accessed_attributes)
    return TypeInfo.from_set(typeinfoset)


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
    return [CallTrace(t) for t in zip(*results)]


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

        # Clear typevar_index from non-homogeneous types: they're being combined
        # into a union and are no longer participating in the generalization pattern.
        def clear_typevar_index(t: TypeInfo) -> TypeInfo:
            class T(TypeInfo.Transformer):
                def visit(vself, node: TypeInfo) -> TypeInfo:
                    return super().visit(node.replace(typevar_index=0))
            return T().visit(t)

        combined = TypeInfo.from_set({clear_typevar_index(t) for t in types})

        # replace type sequence with a variable (but never for parameter lists)
        if (output_options.type_parameters and occurrences[types] > 1 and combined.is_union()
                and not any(isinstance(t, ListTypeInfo) for t in types)):
            if types not in typevars:
                typevars[types] = combined.replace(typevar_index = len(typevars)+1)
            return typevars[types]

        return merged_types(combined.to_set())

    return [rebuild(types) for types in transposed]
