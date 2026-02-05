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
from righttyper.annotation import FuncAnnotation, ModuleVars
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
    overrides: OverriddenFunction|None

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

        if self.overrides and (inline_types := self.overrides.inline_arg_types):
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
                self.overrides.inline_arg_types = inline_types_prime


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
                    if getattr(func_info, attr) != getattr(func_info2, attr):
                        raise ValueError(f"Incompatible {attr} for {func_id.func_name}:\n" +\
                                         f"    {getattr(func_info, attr)}\n" +\
                                         f"    {getattr(func_info2, attr)}"
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


    def collect_annotations(self) -> tuple[dict[CodeId, FuncAnnotation], dict[Filename, ModuleVars]]:
        """Collects function type annotations from the observed types."""

        def mk_annotation(func_info: FuncInfo) -> FuncAnnotation|None:
            traces = func_info.most_common_traces()
            if not traces:
                return None

            parents_arg_types = None
            if func_info.overrides:
                parents_func = func_info.overrides

                if (
                    output_options.ignore_annotations
                    or not (
                        (parents_arg_types := parents_func.inline_arg_types)
                        or (parents_arg_types := get_typeshed_arg_types(parents_func, func_info.args))
                    )
                ):
                    if (
                        (parent_code := parents_func.code_id)
                        and parent_code in self.func_info
                        and (ann := mk_annotation(self.func_info[parent_code]))
                    ):
                        parents_arg_types = tuple(arg[1] for arg in ann.args)

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
                            # by building sets with the parent's types, we prevent arg. type narrowing
                            *((parents_arg_types[i],) if (
                                  parents_arg_types
                                  and len(parents_arg_types) == len(func_info.args)
                                  and parents_arg_types[i] is not None
                                ) else ()
                             )
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

        annotations = {
            func_info.code_id: FuncAnnotation(
                args=[(arg[0], finalize(arg[1])) for arg in annotation.args],
                retval=finalize(annotation.retval),
                varargs=annotation.varargs,
                kwargs=annotation.kwargs,
                variables=[(var[0], finalize(var[1])) for var in annotation.variables]
            )
            for func_info in self.func_info.values()
            if func_info.code_id not in self.wrapper_code_ids
            if (annotation := mk_annotation(func_info)) is not None
        }

        module_vars = {
            filename: ModuleVars([
                (var_name, finalize(merged_types(var_types, for_variable=True)))
                for var_name, var_types in var_dict.items()
            ])
            for filename, var_dict in self.module_variables.items()
        }

        return annotations, module_vars


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
