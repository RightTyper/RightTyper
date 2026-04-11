import ast
import typing
import builtins
from collections import defaultdict, Counter
import collections.abc as abc
from dataclasses import dataclass, field
import logging

from righttyper.options import output_options
from righttyper.logger import logger
from righttyper.generalize import merged_types, generalize
from righttyper.type_transformers import (
    ClearCodeIdT,
    UnionSizeT,
    SelfT,
    NeverSayNeverT,
    NoReturnToNeverT,
    ExcludeTestTypesT,
    GeneratorToIteratorT,
    TypesUnionT,
    DepthLimitT,
    MakePickleableT,
    LoadTypeObjT
)
from righttyper.typeinfo import TypeInfo, TypeInfoArg, NoneTypeInfo, UnknownTypeInfo, CallTrace
from righttyper.righttyper_types import ArgumentName, VariableName, Filename, CodeId
from righttyper.annotation import FuncAnnotation, ModuleVars, TraceDistribution
from righttyper.type_id import PostponedArg0
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

    traces: Counter[CallTrace] = field(default_factory=Counter)

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


    def compute_type_distributions(self) -> list[TraceDistribution] | None:
        """Computes trace-level type frequency distributions.

        Returns None if there is only one distinct trace (no polymorphism).
        Each trace preserves the coordination between argument types and return type.
        Percentages reflect proportions of all observed calls.
        """
        traces = self.most_common_traces()
        if len(traces) <= 1:
            return None

        n_args = len(self.args)
        total = sum(self.traces.values())
        return [
            TraceDistribution(
                args={
                    self.args[i].arg_name: str(trace[i]) if i < len(trace) - 1 else "?"
                    for i in range(n_args)
                },
                retval=str(trace[-1]),
                pct=round(self.traces[trace] / total * 100, 1)
            )
            for trace in traces
        ]


    def transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this object."""
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


        for trace, count in list(self.traces.items()):
            trace_prime = tuple(tr.visit(t) for t in trace)
            # Use identity rather than ==, as only non-essential attributes may have changed
            if any(old is not new for old, new in zip(trace, trace_prime)):
                if logger.level == logging.DEBUG:
                    logger.debug(
                        type(tr).__name__ + " " + self.code_id.func_name +
                        str(tuple(str(t) for t in trace)) +
                        " -> " +
                        str(tuple(str(t) for t in trace_prime))
                    )
                del self.traces[trace]
                self.traces[trace_prime] = count

        for var_name, var_types in list(self.variables.items()):
            self.variables[var_name] = set(tr.visit(t) for t in var_types)


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


    def _propagate_to_parents(self, raw_annotations: dict[CodeId, FuncAnnotation]) -> None:
        """Propagates types between parent and child methods for LSP compliance.

        Each iteration:
        - Upward: widens parent arg/return types from children's observations.
        - Downward: widens children's arg types to match parent
          (LSP contravariance — children must accept at least what the parent accepts).

        Iterates until stable to handle multi-level hierarchies.
        """

        # Build reverse mapping: parent_code_id → [child_code_id, ...]
        parent_to_children: dict[CodeId, list[CodeId]] = defaultdict(list)
        for fi in self.func_info.values():
            for ov in fi.overrides:
                if ov.code_id and ov.code_id.file_name in self.source_to_module_name:
                    parent_to_children[ov.code_id].append(fi.code_id)

        # Phase 1: Merge static arg types from inline annotations / typeshed.
        # Handles all overrides including typeshed-only parents (no code_id).
        # Static arg types are per-child because inline_arg_types is aligned to the
        # child's parameter order (computed by get_parent_arg_types).
        for fi in self.func_info.values():
            ann = raw_annotations.get(fi.code_id)
            if ann is None:
                continue
            arg_names = tuple(name for name, _ in ann.args)
            for ov in fi.overrides:
                parent_fi = self.func_info.get(ov.code_id) if ov.code_id else None
                static_types = (
                    (ov.inline_arg_types if not output_options.ignore_annotations else None)
                    or get_typeshed_arg_types(ov, parent_fi.args if parent_fi else fi.args)
                )
                if not static_types or len(static_types) != len(arg_names):
                    continue
                new_args = list(ann.args)
                ann_changed = False
                for i, st in enumerate(static_types):
                    if st is not None:
                        m = merged_types({new_args[i][1], st})
                        if m is not new_args[i][1]:
                            new_args[i] = (arg_names[i], m)
                            ann_changed = True
                if ann_changed:
                    ann = FuncAnnotation(
                        args=new_args, retval=ann.retval,
                        varargs=ann.varargs, kwargs=ann.kwargs,
                        variables=ann.variables,
                    )
                    raw_annotations[fi.code_id] = ann

        if not parent_to_children:
            return

        def _widen_annotation(
            target: FuncAnnotation | None,
            target_arg_names: tuple[ArgumentName, ...],
            sources: list[FuncAnnotation],
            merge_args: bool = True,
            merge_retval: bool = True,
        ) -> FuncAnnotation | None:
            """Widen target's types from sources.

            merge_args/merge_retval control which parts are widened.
            Returns a new FuncAnnotation if anything changed, None otherwise.
            """
            n_args = len(target_arg_names)
            matching = [s for s in sources if len(s.args) == n_args]
            if not matching:
                return None

            if merge_args:
                new_args = []
                for i in range(n_args):
                    types: set[TypeInfo] = {s.args[i][1] for s in matching}
                    if target is not None:
                        types.add(target.args[i][1])
                    new_args.append((target_arg_names[i], merged_types(types)))
            else:
                new_args = list(target.args) if target else [(n, UnknownTypeInfo) for n in target_arg_names]

            if merge_retval:
                ret_types: set[TypeInfo] = {s.retval for s in matching}
                if target is not None:
                    ret_types.add(target.retval)
                retval = merged_types(ret_types)
            else:
                retval = target.retval if target else UnknownTypeInfo

            new_ann = FuncAnnotation(
                args=new_args,
                retval=retval,
                varargs=target.varargs if target else None,
                kwargs=target.kwargs if target else None,
                variables=target.variables if target else [],
            )
            return new_ann if new_ann != target else None

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
            parent_ann = raw_annotations.get(parent_id)
            if parent_fi.is_abstract:
                merge_args, merge_retval = True, True
            elif not parent_fi.traces:
                merge_args, merge_retval = True, False   # unobserved concrete
            else:
                merge_args, merge_retval = False, True   # observed concrete

            arg_names = tuple(a.arg_name for a in parent_fi.args)
            new_parent = _widen_annotation(
                parent_ann, arg_names, child_anns,
                merge_args=merge_args, merge_retval=merge_retval,
            )
            if new_parent is not None:
                if parent_ann is None:
                    new_parent = FuncAnnotation(
                        args=new_parent.args, retval=new_parent.retval,
                        varargs=parent_fi.varargs, kwargs=parent_fi.kwargs,
                        variables=[],
                    )
                raw_annotations[parent_id] = new_parent

        # Phase 3: Downward — widen children's arg types from parent annotations.
        # A single pass suffices because parent annotations are fixed after Phase 2.
        for parent_id, child_ids in parent_to_children.items():
            parent_ann = raw_annotations.get(parent_id)
            if parent_ann is None:
                continue
            for cid in child_ids:
                child_ann = raw_annotations.get(cid)
                if child_ann is None:
                    continue
                child_arg_names = tuple(name for name, _ in child_ann.args)
                new_child = _widen_annotation(child_ann, child_arg_names, [parent_ann], merge_args=True, merge_retval=False)
                if new_child is not None:
                    raw_annotations[cid] = new_child


    def collect_annotations(self) -> tuple[dict[CodeId, FuncAnnotation], dict[Filename, ModuleVars], dict[CodeId, list[TraceDistribution]]]:
        """Collects function type annotations from the observed types."""

        def mk_annotation(func_info: FuncInfo) -> FuncAnnotation|None:
            traces = func_info.most_common_traces()
            if not traces:
                return None

            if (signature := generalize(traces)) is None:
                logger.info(f"Unable to generalize {func_info.code_id}: inconsistent traces.\n" +
                            f"{[tuple(str(t) for t in s) for s in traces]}")
                return None

            n_sig_args = len(signature) - 1  # last element is the return type

            ann = FuncAnnotation(
                args=[
                    (
                        arg.arg_name,
                        merged_types({
                            signature[i] if i < n_sig_args else UnknownTypeInfo,
                            *((arg.default,) if arg.default is not None else ()),
                        })
                    )
                    for i, arg in enumerate(func_info.args)
                ],
                retval=signature[-1],
                varargs=func_info.varargs,
                kwargs=func_info.kwargs,
                variables=[
                    (var_name, merged_types(var_types, for_variable=True))
                    for var_name, var_types in func_info.variables.items()
                ]
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
                    str((*(str(arg[1]) for arg in ann.args), str(ann.retval)))
                )
                for var_name, var_type in ann.variables:
                    logger.debug(
                        f"var   {func_info.code_id.func_name} {var_name} {str(var_type)}"
                    )

            return ann

        # In the code below, clone (rather than link to) types from Callable, Generator, etc.,
        # clearing is_self, as it may not apply in the new context.
        def clone(node: TypeInfo) -> TypeInfo:
            class T(TypeInfo.Transformer):
                """Clones the given TypeInfo tree, clearing 'is_self',
                   as the type information may not be equivalent in the new context.
                """
                def visit(vself, node: TypeInfo) -> TypeInfo:
                    return super().visit(node.replace(is_self=False))

            return T().visit(node)

        class ResolvingT(TypeInfo.Transformer):
            """Resolves types that may not be fully known until observed at runtime."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                if node.code_id and (func_info := self.func_info.get(node.code_id)):
                    if (ann := mk_annotation(func_info)):
                        # for Callable, we also merge arguments from annotations with observed ones.
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

                            node = node.replace(args=(
                                TypeInfo.list([
                                    old if not is_unknown(old := get_old_param(i)) else clone(a[1])
                                    for i, a in enumerate(ann.args[int(node.is_bound):])
                                ])
                                if not (func_info.varargs or func_info.kwargs
                                       or any(a.default is not None for a in func_info.args))
                                else
                                ...,
                                old_retval if not is_unknown(old_retval) else clone(ann.retval)
                            ))
                        else:
                            node = clone(ann.retval)

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

                return node

        self.transform_types(ResolvingT())

        # Clear code_id after resolution — it was only needed to look up
        # the function's observations and should not affect type equality.
        # This also deduplicates unions whose members became equal after clearing.
        self.transform_types(ClearCodeIdT())
        self.transform_types(UnionSizeT())

        if output_options.use_typing_self:
            self.transform_types(SelfT())

        if output_options.exclude_test_types:
            self.transform_types(ExcludeTestTypesT(
                self.test_modules,
                detect_by_name=output_options.detect_test_modules_by_name
            ))


        finalizers: list[TypeInfo.Transformer] = []

        def finalize(t: TypeInfo) -> TypeInfo:
            for f in finalizers:
                t_prime = f.visit(t)
                if t is not t_prime:
                    # MakePickleableT just adds noise: omit it
                    if logger.level == logging.DEBUG and type(f) is not MakePickleableT:
                        logger.debug(type(f).__name__ + f" {str(t)} -> {str(t_prime)}")
                    t = t_prime
            return t

        if output_options.type_depth_limit is not None:
            finalizers.append(DepthLimitT(output_options.type_depth_limit))

        if output_options.use_typing_union:
            finalizers.append(TypesUnionT())

        # Only rename to Iterator as a finalizer so that all [Async]Generator arguments
        # are available for generalization
        if output_options.simplify_types:
            finalizers.append(GeneratorToIteratorT())

        # Only rename away from typing.Never now so that list[X]|list[Never] can be simplified
        if not output_options.use_typing_never:
            finalizers.append(NeverSayNeverT())
        else:
            finalizers.append(NoReturnToNeverT())

        finalizers.append(MakePickleableT())

        # Compute type distributions before finalization (uses pre-transformation traces)
        type_distributions: dict[CodeId, list[TraceDistribution]] = {}
        if output_options.type_distribution_comments:
            for func_info in self.func_info.values():
                if func_info.code_id not in self.wrapper_code_ids:
                    if (dist := func_info.compute_type_distributions()):
                        type_distributions[func_info.code_id] = dist

        # Compute raw annotations (before finalization)
        raw_annotations: dict[CodeId, FuncAnnotation] = {
            func_info.code_id: annotation
            for func_info in self.func_info.values()
            if func_info.code_id not in self.wrapper_code_ids
            if (annotation := mk_annotation(func_info)) is not None
        }

        # Propagate child method types up to parent methods
        self._propagate_to_parents(raw_annotations)



        # Apply finalization
        annotations = {
            code_id: FuncAnnotation(
                args=[(arg[0], finalize(arg[1])) for arg in annotation.args],
                retval=finalize(annotation.retval),
                varargs=annotation.varargs,
                kwargs=annotation.kwargs,
                variables=[(var[0], finalize(var[1])) for var in annotation.variables]
            )
            for code_id, annotation in raw_annotations.items()
        }

        module_vars = {
            filename: ModuleVars([
                (var_name, finalize(merged_types(var_types, for_variable=True)))
                for var_name, var_types in var_dict.items()
            ])
            for filename, var_dict in self.module_variables.items()
        }

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
    return tuple(
        t.visit(arg) if arg is not None else None
        for arg in args
    )
