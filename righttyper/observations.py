import ast
import typing
from typing import cast
import builtins
from collections import defaultdict, Counter
import collections.abc as abc
from dataclasses import dataclass, field
import logging

from righttyper.options import output_options
from righttyper.logger import logger
from righttyper.generalize import merged_types, generalize, is_homogeneous
from righttyper.type_transformers import (
    UnionSizeT,
    NeverSayNeverT,
    NoReturnToNeverT,
    ExcludeTestTypesT,
    GeneratorToIteratorT,
    TypesUnionT,
    DepthLimitT,
    MakePickleableT,
    LoadTypeObjT
)
from righttyper.typeinfo import TypeInfo, TypeInfoArg, UnknownTypeInfo, UnionTypeInfo, CallTrace
from righttyper.righttyper_types import ArgumentName, VariableName, Filename, CodeId
from righttyper.annotation import FuncAnnotation, ModuleVars, TraceDistribution
from righttyper.type_id import PostponedArg0, get_type_name
from righttyper.typeshed import get_typeshed_func_params


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    default: TypeInfo|None


@dataclass
class OverriddenFunction:
    """Describes an overridden function."""
    module: str
    qualname: str
    code_id: CodeId|None
    inline_arg_types: tuple[TypeInfo|None, ...]|None = None


@dataclass(eq=True)
class FuncInfo:
    code_id: CodeId
    args: tuple[ArgInfo, ...]
    varargs: ArgumentName|None
    kwargs: ArgumentName|None
    overrides: list[OverriddenFunction] = field(default_factory=list)

    is_abstract: bool = False

    # The defining class of the method, if this is a method.  Set during
    # registration from get_self_type's self_replacement.
    defining_class: TypeInfo | None = None

    traces: Counter[CallTrace] = field(default_factory=Counter)

    # Per-variable observed types.  The recorder also injects a constructor
    # ceiling: `p = Foo(...)` adds `Foo` (the resolved callee type) to the
    # set, so `merged_types`'s lub naturally subsumes runtime subtypes
    # (e.g. PosixPath ⊆ Path → Path).  The synthetic key `'<return>'` carries
    # the same kind of ceiling for the function's return value (extracted
    # by mk_annotation and lubbed into the retval).
    # TODO ideally the variables should be included in the trace, so that they can be filtered
    # and also included in any type patterns.
    variables: dict[VariableName, set[TypeInfo]] = field(default_factory=lambda: defaultdict(set))


    def most_common_traces(self) -> list[CallTrace]:
        """Returns the top X% most common call traces, turning type checking into anomaly detection."""
        if output_options.use_top_pct == 100:
            return list(self.traces)

        threshold = sum(self.traces.values()) * output_options.use_top_pct / 100
        cumulative = 0

        traces = list()
        for trace, count in self.traces.most_common():
            if cumulative >= threshold:
                break
            cumulative += count
            traces.append(trace)

        return traces


    def compute_type_distributions(
        self,
        resolve: typing.Callable[[TypeInfo], TypeInfo] | None = None,
    ) -> list[TraceDistribution] | None:
        """Computes trace-level type frequency distributions.

        Returns None if there is only one distinct trace (no polymorphism).
        Each trace preserves the coordination between argument types and return type.
        Percentages reflect proportions of all observed calls.
        If resolve is given, it is applied to each TypeInfo before rendering.
        """
        traces = self.most_common_traces()
        if len(traces) <= 1:
            return None

        def fmt(t: TypeInfo) -> str:
            return str(resolve(t) if resolve else t)

        n_args = len(self.args)
        total = sum(self.traces.values())
        return [
            TraceDistribution(
                args={
                    self.args[i].arg_name: fmt(trace[i]) if i < len(trace) - 1 else "?"
                    for i in range(n_args)
                },
                retval=fmt(trace[-1]),
                pct=round(self.traces[trace] / total * 100, 1)
            )
            for trace in traces
        ]


    def transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this object."""
        if self.defining_class is not None:
            new_dc = tr.visit(self.defining_class)
            if new_dc is not self.defining_class:
                self.defining_class = new_dc

        args_prime = tuple(
            ArgInfo(
                arg.arg_name,
                tr.visit(arg.default) if arg.default is not None else None
            )
            for arg in self.args
        )

        if any(
            old_arg.default is not new_arg.default
            for old_arg, new_arg in zip(self.args, args_prime)
        ):
            if logger.level == logging.DEBUG:
                logger.debug(
                    type(tr).__name__ + " " + self.code_id.func_name +
                    str(tuple(str(arg.default) for arg in self.args)) +
                    " -> " +
                    str(tuple(str(arg.default) for arg in args_prime))
                )
            self.args = args_prime

        for override in self.overrides:
            if (inline_types := override.inline_arg_types):
                inline_types_prime = tuple(
                    tr.visit(it) if it else None
                    for it in inline_types
                )

                if any(
                    old_it is not new_it
                    for old_it, new_it in zip(inline_types, inline_types_prime)
                ):
                    if logger.level == logging.DEBUG:
                        logger.debug(
                            type(tr).__name__ + " " + self.code_id.func_name +
                            str(tuple(str(it) for it in inline_types)) +
                            " -> " +
                            str(tuple(str(it) for it in inline_types_prime))
                        )
                    override.inline_arg_types = inline_types_prime


        for trace in list(self.traces):
            # Re-read count from the live counter: an earlier iteration may have
            # transformed another trace's content to match this key, contributing
            # additional count we must carry through this iteration's rebuild.
            count = self.traces[trace]
            old_fac = trace.first_arg_class
            new_fac = tr.visit(old_fac) if old_fac is not None else None
            trace_prime = CallTrace(
                [tr.visit(t) for t in trace],
                first_arg_class=new_fac,
            )
            # Use identity rather than ==, as only non-essential attributes may have changed
            if any(old is not new for old, new in zip(trace, trace_prime)) or old_fac is not new_fac:
                if logger.level == logging.DEBUG:
                    logger.debug(
                        type(tr).__name__ + " " + self.code_id.func_name +
                        str(tuple(str(t) for t in trace)) +
                        " -> " +
                        str(tuple(str(t) for t in trace_prime))
                    )
                del self.traces[trace]
                self.traces[trace_prime] += count

        for var_name, var_types in list(self.variables.items()):
            self.variables[var_name] = set(tr.visit(t) for t in var_types)


def _stamp_self(
    per_trace: list[TypeInfo],
    self_classes: list[TypeInfo],
) -> TypeInfo | list[TypeInfo]:
    """Replace nodes that are leaf-equal to the per-trace self_class in EVERY
    trace with `typing.Self`.

    Returns a single TypeInfo when the position should be uniformly replaced
    with Self, or a per-trace list of TypeInfos when only nested positions
    were replaced (or nothing was replaced).

    `self_classes[i]` is the receiver TypeInfo for trace i. Comparison uses
    TypeInfo equality (structural: module + name) rather than `type_obj is`,
    so this works even when type_obj reloads as None from a serialized .rt
    file.
    """
    # Leaf match: every trace has matching TypeInfo and no args.
    if all(t == sc and not t.args
           for t, sc in zip(per_trace, self_classes)):
        return TypeInfo.from_type(typing.Self)

    # Recurse if structurally homogeneous across traces.
    if not is_homogeneous(per_trace):
        return list(per_trace)

    first = per_trace[0]
    n_args = len(first.args)
    # For each arg position, gather the per-trace types.
    new_args_per_trace: list[list] = [list(t.args) for t in per_trace]
    for i in range(n_args):
        col = [t.args[i] for t in per_trace]
        if all(isinstance(a, TypeInfo) for a in col):
            sub = _stamp_self([cast(TypeInfo, a) for a in col], self_classes)
            if isinstance(sub, TypeInfo):
                # Uniform replacement at this nested position
                for trace_idx in range(len(per_trace)):
                    new_args_per_trace[trace_idx][i] = sub
            else:
                # Per-trace nested results
                for trace_idx, replaced in enumerate(sub):
                    new_args_per_trace[trace_idx][i] = replaced
        # else: ellipsis or other non-TypeInfo arg — leave as-is
    return [
        per_trace[trace_idx].replace(args=tuple(new_args_per_trace[trace_idx]))
        for trace_idx in range(len(per_trace))
    ]


class _CloneForContextT(TypeInfo.Transformer):
    """Implements `_clone_for_context`. See that function for parameter semantics."""
    def __init__(self,
                 source_self_class: TypeInfo | None,
                 dest_self_class: TypeInfo | None) -> None:
        self.source_self_class = source_self_class
        self.dest_self_class = dest_self_class
        # Self carries through the substitution (rather than being concretized) when
        # source has no self_class, or when source.self_class is the same as or a
        # subclass of dest.self_class — i.e., dest's Self is at least as wide as
        # source's Self in their respective contexts.
        #
        # The subclass branch needs live `type_obj` to walk MRO via issubclass.
        # For `.rt` files where type_obj reloads as None (e.g., classes from
        # __main__ or unimportable modules), this branch silently fails and the
        # Self in source gets substituted to source's concrete class. The
        # annotation is still correct, just less precise (concrete instead of Self).
        self.keep_self = (
            source_self_class is None
            or (dest_self_class is not None and source_self_class == dest_self_class)
            or (
                source_self_class is not None
                and dest_self_class is not None
                and isinstance(source_self_class.type_obj, type)
                and isinstance(dest_self_class.type_obj, type)
                and issubclass(source_self_class.type_obj, dest_self_class.type_obj)
            )
        )
        # Gate Self re-stamping on the same option that gates trace-time
        # _stamp_self: pre-3.11 targets must not see typing.Self injected here.
        self.can_restamp = (
            output_options.use_typing_self
            and dest_self_class is not None
            and (source_self_class is None or source_self_class == dest_self_class)
        )

    def visit(self, n: TypeInfo) -> TypeInfo:
        if n.type_obj is typing.Self:
            if self.keep_self:
                return n
            if self.source_self_class is not None:
                return self.source_self_class
        if self.can_restamp and not n.args and n == self.dest_self_class:
            return TypeInfo.from_type(typing.Self)
        return super().visit(n)


def _clone_for_context(node: TypeInfo,
                       source_self_class: TypeInfo | None = None,
                       dest_self_class: TypeInfo | None = None) -> TypeInfo:
    """Clone a TypeInfo subtree for use in a different annotation context.

    `source_self_class` is the class that `typing.Self` resolves to in the
    annotation `node` came from — i.e., the source annotation's `self_class`.
    None for non-method sources (anonymous functions, lambdas, plain functions).

    `dest_self_class` is the class that `typing.Self` will resolve to in the
    annotation `node` is being cloned into — i.e., the destination annotation's
    `self_class`. None for non-method destinations.

    Both refer to the *meaning of Self* in their respective contexts, not to
    where the underlying code is lexically defined (which is `defining_class`
    on FuncInfo). For a freshly-built annotation `defining_class == self_class`,
    but synthetic annotations (e.g., from `_register_parent_function`) can
    diverge.

    Behavior:
    - Substitute `typing.Self` -> `source_self_class` when leaving the source
      context (Self's meaning is local).
    - With `dest_self_class` set AND source either anonymous or matching dest,
      re-stamp leaf nodes whose TypeInfo equals `dest_self_class` as
      `typing.Self`. This recovers Self when ResolvingT injects concrete types
      that align with the destination's receiver — e.g., a lambda capturing
      self pulled into a method's retval.
    """
    return _CloneForContextT(source_self_class, dest_self_class).visit(node)


class ResolvingT(TypeInfo.Transformer):
    """Resolves TypeInfo nodes that carry a `code_id` by substituting the
    referenced annotation's signature. Also unwraps `PostponedArg0`.

    Used at mk_annotation time (per-trace, with annotations built so far) and
    at collect_annotations time (over each annotation's args/retval). Cross-
    annotation `Self` semantics flow through `_clone_for_context` using the
    source annotation's `self_class` and the destination's `self_class`.
    """
    def __init__(self,
                 annotations: dict["CodeId", "FuncAnnotation"],
                 func_info: dict["CodeId", "FuncInfo"],
                 dest_self_class: TypeInfo | None = None):
        self.annotations = annotations
        self.func_info = func_info
        self.dest_self_class = dest_self_class

    def visit(self, node: TypeInfo) -> TypeInfo:
        pre = node
        node = super().visit(node)

        if node.code_id and (ann := self.annotations.get(node.code_id)):
            cl = lambda t: _clone_for_context(t, ann.self_class, self.dest_self_class)
            if node.type_obj in (abc.Callable, typing.Callable):
                old_params = (
                    node.args[0].args
                    if node.args and isinstance(node.args[0], TypeInfo) and node.args[0].is_list()
                    else ()
                )
                old_retval = node.args[-1] if node.args else UnknownTypeInfo

                def get_old_param(i: int) -> TypeInfoArg:
                    return old_params[i] if i < len(old_params) else UnknownTypeInfo

                def is_unknown(t: TypeInfoArg) -> bool:
                    return isinstance(t, TypeInfo) and t.is_unknown

                # FIXME skip 'self'/'cls' by name rather than assuming it's first
                ann_arg_types = list(ann.args.values())
                if node.is_bound:
                    ann_arg_types = ann_arg_types[1:]

                fi = self.func_info.get(node.code_id)
                use_ann_args = fi is not None and not (
                    ann.varargs or ann.kwargs
                    or any(a.default is not None for a in fi.args)
                )
                node = node.replace(args=(
                    TypeInfo.list([
                        old if not is_unknown(old := get_old_param(i)) else cl(t)
                        for i, t in enumerate(ann_arg_types)
                    ]) if use_ann_args else ...,
                    old_retval if not is_unknown(old_retval) else cl(ann.retval)
                ))
            else:
                node = cl(ann.retval)

        if node.type_obj is PostponedArg0:
            # e.g. PostponedArg0[Iterator[X]] -> X
            node = (
                node.args[0].args[0]
                if node.args
                    and isinstance(node.args[0], TypeInfo)
                    and node.args[0].args
                    and isinstance(node.args[0].args[0], TypeInfo)
                else UnknownTypeInfo
            )

        # Clear code_id once resolved; if children changed, dedup the union
        # so members that became equal after clearing collapse.
        if node.code_id:
            node = node.replace(code_id=None)
        if pre is not node and isinstance(node, UnionTypeInfo):
            node = TypeInfo.from_set(node.to_set(), typevar_index=node.typevar_index)

        return node


class NormalizeSubtypeT(TypeInfo.Transformer):
    """Replace TypeInfos whose type_obj is a strict subtype of `defining_class`
    with `defining_class` itself (inheritance normalization).

    For single-observation cases this widens the annotation toward the method's
    defining class (more permissive) instead of leaving the observed subclass.
    For multi-trace cases, lub's subtype merge produces the same result, so
    this transformer only changes single-observation outputs.

    Example. Given:

        class A:
            def merge(self, other): ...
        class B(A): ...
        A().merge(B())

    The single observed call has `other: B`. `B` is a strict subtype of
    `A` (the defining class of `merge`), so without this transformer the
    annotation is `def merge(self, other: "B")` — fidelity to the
    observation but tighter than the method's actual contract. With it,
    `B` normalizes to `A`, giving `def merge(self, other: "A")`. With
    multiple observations Self detection would already produce
    `other: Self`; this transformer plugs the single-observation gap.
    """
    def __init__(self, defining_class: type):
        self.defining_class = defining_class

    def visit(self, node: TypeInfo) -> TypeInfo:
        if (hasattr(node.type_obj, "__mro__")
                and self.defining_class in cast(type, node.type_obj).__mro__
                and node.type_obj is not self.defining_class):
            node = get_type_name(self.defining_class)
        return super().visit(node)


class Observations:
    def __init__(self) -> None:
        # Visited functions and information about them
        self.func_info: dict[CodeId, FuncInfo] = dict()

        # per-module variables
        self.module_variables: dict[Filename, dict[VariableName, set[TypeInfo]]] = defaultdict(lambda: defaultdict(set))

        # Mapping of sources to their module names
        self.source_to_module_name: dict[Filename, str] = {}

        # Set of test modules
        self.test_modules: set[str] = set()

        # CodeIds of wrapper functions (with __wrapped__) — skip annotation
        self.wrapper_code_ids: set[CodeId] = set()

        # Canonical type name map: (original_module, original_name) → (canonical_module, canonical_name).
        # Set by recorder from TypeMap. Used in collect_annotations to fix names
        # of types introduced by lub (e.g., pathlib._local.Path → pathlib.Path).
        # All strings, serializable for .rt files.
        self.type_name_map: dict[tuple[str, str], tuple[str, str]] = {}

        # Per-function accessed attributes, gathered by static analysis of the
        # source in the loader.  Maps CodeId → {variable_name → {attr, ...}}.
        # Serialized in .rt files so that the process step can use them for
        # attribute-based type simplification (lub Rule 7/8).
        self.accessed_attributes: dict[CodeId, dict[str, set[str]]] = {}

        # Per-file accessed attributes for module-level globals only.
        # Precomputed during the run step (where co_varnames is available to
        # filter out function-local variables) and serialized for process.
        self.module_accessed_attributes: dict[Filename, dict[str, set[str]]] = {}


    def transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this class."""

        for func_info in self.func_info.values():
            func_info.transform_types(tr)

        for var_dict in list(self.module_variables.values()):
            for var_name, var_types in list(var_dict.items()):
                var_dict[var_name] = set(tr.visit(t) for t in var_types)


    def merge_observations(self, obs2: "Observations") -> None:
        """Merges other observations into this one."""

        for func_id, func_info2 in obs2.func_info.items():
            if (func_info := self.func_info.get(func_id)):
                for attr in ('varargs', 'kwargs', 'overrides'):
                    v1, v2 = getattr(func_info, attr), getattr(func_info2, attr)
                    if v1 != v2:
                        if attr == 'overrides':
                            # 'overrides' is semantically a SET (the consumers in
                            # _propagate_to_parents iterate it without depending
                            # on order, and look up each entry by code_id), so it
                            # should arguably be one — left as a list for now to
                            # avoid touching every recording site.
                            #
                            # Two FuncInfo entries for the same code_id can have
                            # different overrides because _register_parent_function
                            # early-exits its MRO walk at the first ancestor that
                            # is already registered, so the recorded chain depends
                            # on observation order.  Both lists are valid partial
                            # views of the same MRO; merging them as a union (deduped
                            # by code_id) yields the most complete view available.
                            seen: set[CodeId | None] = {ov.code_id for ov in v1}
                            merged_overrides = list(v1)
                            for ov in v2:
                                if ov.code_id not in seen:
                                    merged_overrides.append(ov)
                                    seen.add(ov.code_id)
                            func_info.overrides = merged_overrides
                        else:
                            raise ValueError(f"Incompatible {attr} for {func_id.func_name}:\n" +\
                                             f"    {v1}\n" +\
                                             f"    {v2}"
                            )

                # Merge args, unioning default types if they differ
                args1, args2 = func_info.args, func_info2.args
                if len(args1) != len(args2):
                    raise ValueError(f"Incompatible args length for {func_id.func_name}:\n" +\
                                     f"    {args1}\n" +\
                                     f"    {args2}"
                    )
                for a1, a2 in zip(args1, args2):
                    if a1.arg_name != a2.arg_name:
                        raise ValueError(f"Incompatible arg names for {func_id.func_name}:\n" +\
                                         f"    {a1.arg_name}\n" +\
                                         f"    {a2.arg_name}"
                        )
                if args1 != args2:
                    func_info.args = tuple(
                        ArgInfo(
                            a1.arg_name,
                            TypeInfo.from_set({d for d in (a1.default, a2.default) if d is not None}, empty_is_none=True)
                        )
                        for a1, a2 in zip(args1, args2)
                    )

                func_info.traces.update(func_info2.traces)

                for varname, typeset in func_info2.variables.items():
                    func_info.variables[varname] |= typeset
            else:
                self.func_info[func_id] = func_info2

        for filename, var_dict2 in obs2.module_variables.items():
            if (var_dict := self.module_variables.get(filename)):
                for varname, typeset in var_dict2.items():
                    var_dict[varname] |= typeset
            else:
                self.module_variables[filename] = var_dict2

        self.source_to_module_name |= obs2.source_to_module_name
        self.test_modules |= obs2.test_modules
        self.wrapper_code_ids |= obs2.wrapper_code_ids
        self.type_name_map |= obs2.type_name_map
        self.accessed_attributes.update(obs2.accessed_attributes)
        self.module_accessed_attributes.update(obs2.module_accessed_attributes)


    def _propagate_to_parents(self, raw_annotations: dict[CodeId, FuncAnnotation]) -> None:
        """Propagates types between parent and child methods for LSP compliance.

        - Upward: widens parent arg/return types from children's observations.
        - Downward: widens children's arg types to match parent
          (LSP contravariance — children must accept at least what the parent accepts).

        A single pass in each direction suffices because overrides are recorded
        for all ancestors (not just the immediate parent), so parent_to_children
        is a flat mapping from every ancestor to each of its descendants.
        """

        # Functions that override a parent method — typically a small subset.
        overriding = [fi for fi in self.func_info.values() if fi.overrides]

        # Build reverse mapping: parent_code_id → [child_code_id, ...]
        parent_to_children: dict[CodeId, list[CodeId]] = defaultdict(list)
        for fi in overriding:
            for ov in fi.overrides:
                if ov.code_id and ov.code_id.file_name in self.source_to_module_name:
                    parent_to_children[ov.code_id].append(fi.code_id)

        # Phase 1: Merge static arg types from inline annotations / typeshed.
        # Handles all overrides including typeshed-only parents (no code_id).
        # Static arg types are per-child because inline_arg_types is aligned to the
        # child's parameter order (computed by get_parent_arg_types).
        for fi in overriding:
            ann = raw_annotations.get(fi.code_id)
            if ann is None:
                continue
            for ov in fi.overrides:
                # Skip if parent is observed — Phases 2/3 will merge its actual types.
                if ov.code_id and ov.code_id in raw_annotations:
                    continue
                parent_fi = self.func_info.get(ov.code_id) if ov.code_id else None
                static_types = (
                    (ov.inline_arg_types if not output_options.ignore_annotations else None)
                    or get_typeshed_arg_types(ov, parent_fi.args if parent_fi else fi.args)
                )
                if not static_types or len(static_types) != len(ann.args):
                    continue
                for (name, existing), st in zip(ann.args.items(), static_types):
                    if st is not None:
                        m = merged_types({existing, st})
                        if m is not existing:
                            ann.args[name] = m

        if not parent_to_children:
            return

        def _widen_annotation(
            target: FuncAnnotation,
            sources: list[FuncAnnotation],
            merge_args: bool = True,
            merge_retval: bool = True,
        ) -> None:
            """Widen target's types from sources, mutating target in place.

            Self-handling for cross-class merges:

            - **Source types are cloned** via `_clone_for_context(t,
              source.self_class, target.self_class)`. Inside clone, `typing.Self`
              is substituted to source's concrete class UNLESS source's
              self_class is the same as or a subclass of target's (parent's
              Self is at least as wide as child's, so it stays as Self).

            - **Target's own types are added as-is** (no clone) — they're
              already in destination's context.

            arg0 (self/cls) is skipped only when target's arg0 is already
            non-Unknown — receiver type is fixed by target.self_class. When
            target's arg0 is Unknown (synthetic parent with no traces),
            widening fills it in from children; the keep_self rule in clone
            preserves Self from children's arg0.
            """
            matching = [s for s in sources if len(s.args) == len(target.args)]
            if not matching:
                return

            if merge_args:
                for i, name in enumerate(target.args):
                    # Skip arg0 (self/cls) for methods that already have a known
                    # receiver type: it's always exactly target.self_class (or
                    # Self), not subject to Liskov widening from parents. If
                    # target's arg0 is still Unknown (synthetic parent with no
                    # traces), let widening fill it in from children — the
                    # keep_self branch in _clone_for_context preserves Self
                    # from children's arg0 when source.self_class is a subclass
                    # of target.self_class.
                    if (i == 0 and target.self_class is not None
                            and not target.args[name].is_unknown):
                        continue
                    types: set[TypeInfo] = {
                        _clone_for_context(list(s.args.values())[i],
                                           s.self_class, target.self_class)
                        for s in matching
                    }
                    if not target.args[name].is_unknown:
                        # target's own type is already in the destination context;
                        # add it directly without cloning (no substitution, no restamp).
                        types.add(target.args[name])
                    target.args[name] = merged_types(types)

            if merge_retval:
                ret_types: set[TypeInfo] = {
                    _clone_for_context(s.retval, s.self_class, target.self_class)
                    for s in matching
                }
                if not target.retval.is_unknown:
                    ret_types.add(target.retval)
                target.retval = merged_types(ret_types)

        # Phase 2: Upward — propagate children's types to parents.
        for parent_id, child_ids in parent_to_children.items():
            child_anns = [raw_annotations[cid] for cid in child_ids if cid in raw_annotations]
            if not child_anns:
                continue

            parent_fi = self.func_info.get(parent_id)
            if parent_fi is None:
                continue

            #   Args:    only for unobserved/abstract (child can widen without affecting parent)
            #   Returns: only for observed/abstract (LSP covariance); not for unobserved
            #            concrete (body could contradict)
            if parent_fi.is_abstract:
                merge_args, merge_retval = True, True
            elif not parent_fi.traces:
                merge_args, merge_retval = True, False   # unobserved concrete
            else:
                merge_args, merge_retval = False, True   # observed concrete

            if parent_id not in raw_annotations:
                raw_annotations[parent_id] = FuncAnnotation(
                    args={a.arg_name: UnknownTypeInfo for a in parent_fi.args},
                    retval=UnknownTypeInfo,
                    varargs=parent_fi.varargs, kwargs=parent_fi.kwargs,
                    variables={},
                    self_class=parent_fi.defining_class,
                )

            _widen_annotation(
                raw_annotations[parent_id], child_anns,
                merge_args=merge_args, merge_retval=merge_retval,
            )

        # Phase 3: Downward — widen children's arg types from parent annotations.
        # A single pass suffices because parent annotations are fixed after Phase 2.
        for parent_id, child_ids in parent_to_children.items():
            parent_ann = raw_annotations.get(parent_id)
            if parent_ann is None:
                continue
            for cid in child_ids:
                if (child_ann := raw_annotations.get(cid)):
                    _widen_annotation(child_ann, [parent_ann], merge_args=True, merge_retval=False)


    @staticmethod
    def mk_annotation(func_info: FuncInfo,
                      annotations: dict[CodeId, FuncAnnotation] | None = None,
                      func_info_map: dict[CodeId, FuncInfo] | None = None,
                      accessed_attributes: dict[str, set[str]] | None = None,
                      ) -> FuncAnnotation|None:
        traces = func_info.most_common_traces()
        if not traces:
            return None

        # Resolve code_id references in traces using already-built annotations.
        # Topo sort guarantees dependencies are built first, so by the time we
        # process this trace, any nested Callable/Generator code_ids it contains
        # have ready annotations. This lets the Self detection below recurse
        # into resolved types (e.g., a lambda's retval that matches self_class).
        if annotations:
            resolver = ResolvingT(
                annotations,
                func_info_map or {},
                dest_self_class=func_info.defining_class,
            )
            traces = [
                CallTrace(
                    [resolver.visit(t) for t in trace],
                    first_arg_class=trace.first_arg_class,
                )
                for trace in traces
            ]

        # Self detection: if this is a method, structurally stamp typing.Self
        # at positions where the per-trace receiver class matches across all
        # traces. arg0 and retval get single-trace detection; other args
        # require >= 2 traces. After stamping, normalize remaining subtypes
        # of the defining class to defining_class itself.
        if ((defining_class := func_info.defining_class) is not None
                and isinstance(defining_class.type_obj, type)):
            n_positions = len(traces[0])
            modified = [list(t) for t in traces]

            # Stamp typing.Self only when the output target supports it (3.11+).
            # For methods, every trace carries first_arg_class (set in
            # PendingCallTrace.finish() at record time).
            if output_options.use_typing_self:
                self_classes = [cast(TypeInfo, t.first_arg_class) for t in traces]
                for pos in range(n_positions):
                    is_arg0 = (pos == 0)
                    is_retval = (pos == n_positions - 1)
                    if not (is_arg0 or is_retval) and len(traces) < 2:
                        continue  # don't stamp Self on non-arg0/retval from single observation
                    col = [m[pos] for m in modified]
                    if not all(isinstance(c, TypeInfo) for c in col):
                        continue
                    result = _stamp_self([cast(TypeInfo, c) for c in col], self_classes)
                    if isinstance(result, TypeInfo):
                        for m in modified:
                            m[pos] = result
                    else:
                        for i, replaced in enumerate(result):
                            modified[i][pos] = replaced

            # Inheritance widening: replace subtypes of defining_class with defining_class.
            # Applies regardless of typing.Self availability.
            normalize = NormalizeSubtypeT(cast(type, defining_class.type_obj))
            modified = [
                [normalize.visit(c) if isinstance(c, TypeInfo) else c for c in m]
                for m in modified
            ]

            traces = [
                CallTrace(m, first_arg_class=t.first_arg_class)
                for m, t in zip(modified, traces)
            ]

        if (signature := generalize(traces)) is None:
            logger.info(f"Unable to generalize {func_info.code_id}: inconsistent traces.\n" +
                        f"{[tuple(str(t) for t in s) for s in traces]}")
            return None

        n_sig_args = len(signature) - 1  # last element is the return type

        # Pull the return-ceiling set off `func_info.variables` (synthetic
        # `<return>` key populated by the recorder). It will be lubbed into
        # the retval below.
        from righttyper.recorder import RETURN_CEILING_KEY
        return_ceilings = func_info.variables.get(RETURN_CEILING_KEY, set())

        # Resolve code_ids in variable types too (they don't go through traces).
        variables = {
            var_name: merged_types(
                {resolver.visit(t) for t in var_types} if annotations else var_types,
                for_variable=True,
                accessed_attributes=accessed_attributes.get(var_name) if accessed_attributes else None,
            )
            for var_name, var_types in func_info.variables.items()
            if var_name != RETURN_CEILING_KEY
        }

        retval = (
            merged_types({signature[-1], *return_ceilings})
            if return_ceilings else signature[-1]
        )

        ann = FuncAnnotation(
            args={
                arg.arg_name:
                    merged_types(
                        {
                            signature[i] if i < n_sig_args else UnknownTypeInfo,
                            *((arg.default,) if arg.default is not None else ()),
                        },
                        accessed_attributes=accessed_attributes.get(arg.arg_name) if accessed_attributes else None,
                    )
                for i, arg in enumerate(func_info.args)
            },
            retval=retval,
            varargs=func_info.varargs,
            kwargs=func_info.kwargs,
            variables=variables,
            self_class=func_info.defining_class,
        )

        if logger.level == logging.DEBUG:
            for trace, count in list(func_info.traces.items()):
                logger.debug(
                    "trace " + func_info.code_id.func_name +
                    str(tuple(str(t) for t in trace)) +
                    f" {count}x"
                )
            logger.debug(
                "ann   " + func_info.code_id.func_name +
                str((*(str(t) for t in ann.args.values()), str(ann.retval)))
            )
            for var_name, var_type in ann.variables.items():
                logger.debug(
                    f"var   {func_info.code_id.func_name} {var_name} {str(var_type)}"
                )

        return ann


    def code_id_topo_sort(self) -> list[CodeId]:
        """Topologically sort functions, so as to compute annotations in dependency order."""
        visited = set()
        result = []

        def walk(t: TypeInfo) -> None:
            if t.code_id:
                visit(t.code_id)
            for a in t.args:
                if isinstance(a, TypeInfo):
                    walk(a)

        def visit(code_id: CodeId) -> None:
            if code_id not in visited:
                visited.add(code_id)
                if (func_info := self.func_info.get(code_id)):
                    for tr in func_info.traces:
                        for t in tr:
                            walk(t)
                    for var_types in func_info.variables.values():
                        for t in var_types:
                            walk(t)
                    for arg in func_info.args:
                        if arg.default is not None:
                            walk(arg.default)
                    for override in func_info.overrides:
                        if override.inline_arg_types:
                            for it in override.inline_arg_types:
                                if it is not None:
                                    walk(it)
                    result.append(code_id)

        for code_id in self.func_info.keys():
            visit(code_id)

        return result


    def collect_annotations(self) -> tuple[dict[CodeId, FuncAnnotation], dict[Filename, ModuleVars], dict[CodeId, list[TraceDistribution]]]:
        """Collects function type annotations from the observed types."""

        accessed_attrs = self.accessed_attributes

        module_accessed = self.module_accessed_attributes

        annotations: dict[CodeId, FuncAnnotation] = {}
        for code_id in self.code_id_topo_sort():
            annotation = Observations.mk_annotation(
                self.func_info[code_id],
                annotations=annotations,
                func_info_map=self.func_info,
                accessed_attributes=accessed_attrs.get(code_id),
            )
            if annotation is not None:
                annotations[code_id] = annotation

        # Merge module variables into ModuleVars (parallel to annotations for functions).
        # Module-scope vars don't go through mk_annotation, so resolve any code_id
        # references against the just-built annotations here.
        mv_resolver = ResolvingT(annotations, self.func_info)
        module_vars: dict[Filename, ModuleVars] = {
            filename: ModuleVars({
                var_name: mv_resolver.visit(merged_types(
                    var_types,
                    for_variable=True,
                    accessed_attributes=module_accessed.get(filename, {}).get(var_name) or None,
                ))
                for var_name, var_types in var_dict.items()
            })
            for filename, var_dict in self.module_variables.items()
        }

        def _visit_dict(d: dict, tr: TypeInfo.Transformer, tr_name: str, context: str) -> None:
            """Apply tr to each TypeInfo value in d, updating in place with debug logging."""
            for name, t in d.items():
                t_prime = tr.visit(t)
                if t is not t_prime:
                    if logger.level == logging.DEBUG:
                        logger.debug(f"{tr_name} {context}{name}: {t} -> {t_prime}")
                    d[name] = t_prime

        def transform_types(tr: TypeInfo.Transformer) -> None:
            """Applies the 'tr' transformer to all TypeInfo objects."""
            tr_name = type(tr).__name__

            for code_id, annotation in annotations.items():
                prefix = f"{code_id.func_name} "
                _visit_dict(annotation.args, tr, tr_name, prefix)

                t_prime = tr.visit(annotation.retval)
                if t_prime is not annotation.retval:
                    if logger.level == logging.DEBUG:
                        logger.debug(f"{tr_name} {prefix}retval: {annotation.retval} -> {t_prime}")
                    annotation.retval = t_prime

                _visit_dict(annotation.variables, tr, tr_name, prefix)

                if annotation.self_class is not None:
                    sc_prime = tr.visit(annotation.self_class)
                    if sc_prime is not annotation.self_class:
                        if logger.level == logging.DEBUG:
                            logger.debug(f"{tr_name} {prefix}self_class: {annotation.self_class} -> {sc_prime}")
                        annotation.self_class = sc_prime

            for filename, mv in module_vars.items():
                module = self.source_to_module_name.get(filename, filename)
                _visit_dict(mv.variables, tr, tr_name, f"{module} ")


        # Resolve code_id refs unreachable from topo sort (cycles, self-references).
        transform_types(ResolvingT(annotations, self.func_info))

        # Compute type distributions from traces, resolving code_id references
        # so that Callable/generator types render with their actual signatures.
        type_distributions: dict[CodeId, list[TraceDistribution]] = {}
        if output_options.type_distribution_comments:
            resolve = ResolvingT(annotations, self.func_info).visit
            for func_info in self.func_info.values():
                if func_info.code_id not in self.wrapper_code_ids:
                    if (dist := func_info.compute_type_distributions(resolve)):
                        type_distributions[func_info.code_id] = dist

        # Drop wrapper annotations — they were only needed for ResolvingT lookups.
        for cid in self.wrapper_code_ids:
            annotations.pop(cid, None)

        # Propagate child method types up to parent methods
        self._propagate_to_parents(annotations)

        # Apply finalizers (order matters — see comments for rationale)
        if output_options.type_depth_limit is not None:
            transform_types(DepthLimitT(output_options.type_depth_limit))

        if output_options.use_typing_union:
            transform_types(TypesUnionT())

        # Only rename to Iterator as a finalizer so that all [Async]Generator arguments
        # are available for generalization
        if output_options.simplify_types:
            transform_types(GeneratorToIteratorT())

        # Only rename away from typing.Never now so that list[X]|list[Never] can be simplified
        if not output_options.use_typing_never:
            transform_types(NeverSayNeverT())
        else:
            transform_types(NoReturnToNeverT())

        # Exclude test types before capping union size: removing test-module
        # members from unions can bring their size below max_union_size.
        if output_options.exclude_test_types:
            transform_types(ExcludeTestTypesT(
                self.test_modules,
                detect_by_name=output_options.detect_test_modules_by_name
            ))

        transform_types(UnionSizeT())

        # Fix type names introduced by lub (e.g., MRO supertypes
        # with internal module paths like pathlib._local.Path → pathlib.Path).
        if self.type_name_map:
            class _AdjustNewTypeNames(TypeInfo.Transformer):
                """Remap type names using the serializable name map."""
                def visit(vself, node: TypeInfo) -> TypeInfo:
                    key = (node.module, node.name)
                    if (canonical := self.type_name_map.get(key)):
                        if canonical != key:
                            node = node.replace(module=canonical[0], name=canonical[1])
                    return super().visit(node)
            transform_types(_AdjustNewTypeNames())

        transform_types(MakePickleableT())

        return annotations, module_vars, type_distributions


class LoadAndCheckTypesT(LoadTypeObjT):
    """Looks up the type_obj of all types; if not found, transforms it into UnknownTypeInfo."""

    def visit(self, node: TypeInfo) -> TypeInfo:
        node = super().visit(node)
        if node.type_obj is None:
            return UnknownTypeInfo

        return node

                
def get_typeshed_arg_types(
    parents_func: OverriddenFunction,
    child_args: tuple[ArgInfo, ...]
) -> tuple[TypeInfo|None, ...] | None:
    """Returns typeshed type annotations for a parent's method's arguments."""

    module = parents_func.module if parents_func.module else 'builtins'
    if not (args := get_typeshed_func_params(module, parents_func.qualname)):
        return None

    t = LoadAndCheckTypesT()
    # `signature[-1]` is the return type; this consumer wants only args.
    return tuple(
        t.visit(arg) if arg is not None else None
        for arg in args[:-1]
    )
