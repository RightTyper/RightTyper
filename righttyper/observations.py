import ast
import typeshed_client as typeshed
from types import CodeType, FrameType, FunctionType, GeneratorType
import typing
from typing import Final, Any, overload, cast
import builtins
import inspect
from collections import defaultdict, Counter
import collections.abc as abc
from dataclasses import dataclass, field
import logging
from pathlib import Path

from righttyper.options import options
from righttyper.logger import logger
from righttyper.generalize import merged_types, generalize
from righttyper.variable_capture import code2variables
from righttyper.typemap import AdjustTypeNamesT
from righttyper.type_transformers import (
    SelfT,
    NeverSayNeverT,
    NoReturnToNeverT,
    ExcludeTestTypesT,
    ResolveMocksT,
    GeneratorToIteratorT,
    TypesUnionT,
    DepthLimitT,
    MakePickleableT
)
from righttyper.typeinfo import TypeInfo, NoneTypeInfo, AnyTypeInfo, UnknownTypeInfo
from righttyper.righttyper_types import (
    ArgumentName,
    VariableName,
    CodeId,
    Filename,
    FuncId,
    FrameId,
    FuncAnnotation,
    FunctionName,
    CallTrace,
    ModuleVars
)
from righttyper.righttyper_utils import source_to_module_fqn, get_main_module_fqn
from righttyper.righttyper_runtime import (
    find_function,
    unwrap,
    get_value_type,
    get_type_name,
    hint2type,
    PostponedIteratorArg,
)


def find_co_newlocals():
    # CO_NEWLOCALS, when set within a CodeType's co_flags, indicates that
    # a new "locals" dictionary is created; it roughly indicates a function
    # call, as the new dictionary is created for the new scope.
    import dis
    return next(
        flag
        for flag, name in dis.COMPILER_FLAG_NAMES.items()
        if name == "NEWLOCALS"
    )
CO_NEWLOCALS: Final = find_co_newlocals()


# Singleton used to differentiate from None
NO_OBJECT: Final = object()



# Overloads so we don't have to always write CodeId(id(code)), etc.
@overload
def id(obj: CodeType) -> CodeId: ...
@overload
def id(obj: FrameType) -> FrameId: ...
@overload
def id(obj: object) -> int: ...
def id(obj):
    return builtins.id(obj)


@dataclass
class ArgInfo:
    arg_name: ArgumentName
    default: TypeInfo|None


@dataclass
class FunctionDescriptor:
    """Describes a function by name; stands in for a FunctionType where the function
       is a wrapper_descriptor (or possibly other objects), lacking __module__
    """
    __module__: str
    __qualname__: str


@dataclass(eq=True, frozen=True)
class FuncInfo:
    func_id: FuncId
    args: tuple[ArgInfo, ...]
    varargs: ArgumentName|None
    kwargs: ArgumentName|None
    overrides: FunctionType|FunctionDescriptor|None


@dataclass
class PendingCallTrace:
    arg_info: inspect.ArgInfo
    args: tuple[TypeInfo, ...]
    yields: set[TypeInfo] = field(default_factory=set)
    sends: set[TypeInfo] = field(default_factory=set)
    returns: TypeInfo = NoneTypeInfo
    is_async: bool = False
    is_generator: bool = False
    self_type: TypeInfo | None = None
    self_replacement: TypeInfo | None = None


    def process(self) -> CallTrace:
        retval = self.returns

        if self.is_generator:
            y = TypeInfo.from_set(self.yields)
            s = TypeInfo.from_set(self.sends)

            if self.is_async:
                retval = TypeInfo.from_type(abc.AsyncGenerator, module="typing", args=(y, s))
            else:
                retval = TypeInfo.from_type(abc.Generator, module="typing", args=(y, s, self.returns))
            
        type_data = (*self.args, retval)

        if self.self_type and self.self_replacement:
            self_type = cast(TypeInfo, self.self_type)
            self_replacement = cast(TypeInfo, self.self_replacement)

            class SelfTransformer(TypeInfo.Transformer):
                """Replaces 'self' types with the type of the class that defines them,
                   also setting is_self for possible later replacement with typing.Self."""

                def visit(vself, node: TypeInfo) -> TypeInfo:
                    if (
                        hasattr(node.type_obj, "__mro__")
                        and self_type.type_obj in cast(type, node.type_obj).__mro__
                    ):
                        node = self_replacement.replace(is_self=True)

                    return super().visit(node)


            tr = SelfTransformer()
            type_data = (*(tr.visit(arg) for arg in type_data),)

        return type_data


@dataclass
class Observations:
    # Visited functions' and information about them
    functions_visited: dict[CodeType, FuncInfo] = field(default_factory=dict)

    # Started, but not (yet) completed traces
    pending_traces: dict[CodeType, dict[FrameId, PendingCallTrace]] = field(default_factory=lambda: defaultdict(dict))

    # Variables
    # TODO ideally the variables should be included in the trace, so that they can be filtered
    # and also included in any type patterns.
    variables: dict[CodeType, dict[VariableName, set[TypeInfo]]] = field(
                                                    default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    # Object attributes: class_key -> attr_name -> set[TypeInfo]
    object_attributes: dict[object, dict[VariableName, set[TypeInfo]]] = field(
                                                    default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    # Completed traces
    traces: dict[CodeType, Counter[CallTrace]] = field(default_factory=dict)

    # Mapping of sources to their module names
    # TODO handle cases where modules are loaded more than once, e.g. through pytest
    source_to_module_name: dict[str, str|None] = field(default_factory=dict)

    # target __main__ module globals
    main_globals: dict[str, Any]|None = None


    def record_module(
        self,
        code: CodeType,
        frame: FrameType
    ) -> None:
        if code.co_filename not in self.source_to_module_name:
            if (modname := frame.f_globals.get('__name__', None)):
                if modname == "__main__":
                    modname = get_main_module_fqn()
            else:
                modname = source_to_module_fqn(Path(code.co_filename))

            self.source_to_module_name[code.co_filename] = modname


    def record_function(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo,
        overrides: FunctionType|FunctionDescriptor|None
    ) -> None:
        """Records that a function was visited, along with some details about it."""

        if code not in self.functions_visited:
            arg_names = (
                *(a for a in arg_info.args),
                *((arg_info.varargs,) if arg_info.varargs else ()),
                *((arg_info.keywords,) if arg_info.keywords else ())
            )

            defaults = get_defaults(code, frame)

            self.functions_visited[code] = FuncInfo(
                FuncId(
                    Filename(code.co_filename),
                    code.co_firstlineno,
                    FunctionName(code.co_qualname),
                ),
                tuple(
                    ArgInfo(ArgumentName(name), defaults.get(name))
                    for name in arg_names
                ),
                ArgumentName(arg_info.varargs) if arg_info.varargs else None,
                ArgumentName(arg_info.keywords) if arg_info.keywords else None,
                overrides
            )


    @staticmethod
    def _get_arg_types(arg_info: inspect.ArgInfo) -> tuple[TypeInfo, ...]:
        """Computes the types of the given arguments."""
        return (
            *(get_value_type(arg_info.locals[arg_name]) for arg_name in arg_info.args),
            *(
                (TypeInfo.from_set({
                    get_value_type(val) for val in arg_info.locals[arg_info.varargs]
                }),)
                if arg_info.varargs else ()
            ),
            *(
                (TypeInfo.from_set({
                    get_value_type(val) for val in arg_info.locals[arg_info.keywords].values()
                }),)
                if arg_info.keywords else ()
            )
        )


    def record_start(
        self,
        code: CodeType,
        frame: FrameType,
        arg_info: inspect.ArgInfo
    ) -> None:
        """Records a function start."""

        # print(f"record_start {code.co_qualname} {arg_types}")

        self.record_module(code, frame)

        if not (code.co_flags & CO_NEWLOCALS):
            self.functions_visited[code] = FuncInfo(
                FuncId(
                    Filename(code.co_filename),
                    code.co_firstlineno,
                    FunctionName(code.co_qualname),
                ),
                args=(), varargs=None, kwargs=None, overrides=None
            )
        else:
            self_type, self_replacement, overrides = get_self_type(code, arg_info)
            self.record_function(code, frame, arg_info, overrides)

            self.pending_traces[code][id(frame)] = PendingCallTrace(
                arg_info=arg_info,
                args=self._get_arg_types(arg_info),
                is_async=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_COROUTINE)),
                is_generator=bool(code.co_flags & (inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)),
                self_type=self_type, self_replacement=self_replacement
            )


    def record_yield(self, code: CodeType, frame_id: FrameId, yield_value: Any) -> None:
        """Records a yield."""

        # print(f"record_yield {code.co_qualname}")
        if (per_frame := self.pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            tr.yields.add(get_value_type(yield_value))


    def record_send(self, code: CodeType, frame_id: FrameId, send_value: Any) -> None:
        """Records a send."""

        # print(f"record_send {code.co_qualname}")
        if (per_frame := self.pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            tr.sends.add(get_value_type(send_value))


    def _record_variables(self, code: CodeType, frame: FrameType) -> None:
        """Records variables."""
        # print(f"record_variables {code.co_qualname}")

        if not options.variables or not (codevars := code2variables.get(code)):
            return

        scope_vars = self.variables[typing.cast(CodeType, codevars.scope_code)]
        f_locals = frame.f_locals
        value: Any
        dst: str|None
        for src, dst in codevars.variables.items():
            if (value := f_locals.get(src, NO_OBJECT)) is not NO_OBJECT:
                scope_vars[VariableName(dst)].add(get_value_type(value))

        if codevars.self and (self_obj := f_locals.get(codevars.self)) is not None:
            obj_attrs = self.object_attributes[codevars.class_key]
            for src, dst in codevars.attributes.items():
                if (value := getattr(self_obj, src, NO_OBJECT)) is not NO_OBJECT:
                    type_set = obj_attrs[VariableName(src)]
                    type_set.add(get_value_type(value))
                    if dst: scope_vars[VariableName(dst)] = type_set


    def _record_return_type(self, tr: PendingCallTrace, code: CodeType, ret_type: Any) -> None:
        """Records a pending call trace's return type, finishing the trace."""
        assert tr is not None

        tr.returns = (
            ret_type if ret_type is not None
            else (
                # Generators may still be running, or exit with a GeneratorExit exception; we still
                # want them marked as returning None, so they can be simplified to Iterator
                NoneTypeInfo if tr.is_generator else TypeInfo.from_type(typing.NoReturn)
            )
        )

        if code not in self.traces:
            self.traces[code] = Counter()
        self.traces[code].update((tr.process(),))

        # Resample arguments in case they change during execution (e.g., containers)
        tr.args = self._get_arg_types(tr.arg_info)
        self.traces[code].update((tr.process(),))


    def record_return(self, code: CodeType, frame: FrameType, return_value: Any) -> bool:
        """Records a return."""

        # print(f"record_return {code.co_qualname}")
        frame_id = id(frame)
        if (per_frame := self.pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code, get_value_type(return_value))
            self._record_variables(code, frame)
            del per_frame[frame_id]
            return True # found it
        elif code in self.functions_visited:
            self._record_variables(code, frame)

        return False


    def record_no_return(self, code: CodeType, frame: FrameType) -> bool:
        """Records the lack of a return (e.g., because an exception was raised)."""

        # print(f"record_no_return {code.co_qualname}")
        frame_id = id(frame)
        if (per_frame := self.pending_traces.get(code)) and (tr := per_frame.get(frame_id)):
            self._record_return_type(tr, code, None)
            self._record_variables(code, frame)
            del per_frame[frame_id]
            return True # found it
        elif code in self.functions_visited:
            self._record_variables(code, frame)

        return False


    def _transform_types(self, tr: TypeInfo.Transformer) -> None:
        """Applies the 'tr' transformer to all TypeInfo objects in this class."""

        for code, trace_counter in self.traces.items():
            for trace, count in list(trace_counter.items()):
                trace_prime = tuple(tr.visit(t) for t in trace)
                # Use identity rather than ==, as only non-essential attributes may have changed
                if any(old is not new for old, new in zip(trace, trace_prime)):
                    if logger.level == logging.DEBUG:
                        func_info = self.functions_visited.get(code, None)
                        logger.debug(
                            type(tr).__name__ + " " +
                            (func_info.func_id.func_name if func_info else "?") +
                            str(tuple(str(t) for t in trace)) +
                            " -> " +
                            str(tuple(str(t) for t in trace_prime))
                        )
                    del trace_counter[trace]
                    trace_counter[trace_prime] = count

        # TODO how can self.variables change size during iteration?
        for code, var_dict in list(self.variables.items()):
            for var_name, var_types in list(var_dict.items()):
                var_dict[var_name] = set(tr.visit(t) for t in var_types)


    def try_close_generators(self) -> None:
        """Attempts to close any generators that may still be running."""
        pending_generators = set()
        for code, per_frame in self.pending_traces.items():
            if any(tr.is_generator for tr in per_frame.values()):
                pending_generators.add(code)

        for code in pending_generators:
            import gc
            for obj in gc.get_referrers(code):
                # In Python 3.13+, close() doesn't generate a PY_UNWIND
                # https://github.com/python/cpython/issues/140373
                if isinstance(obj, GeneratorType):
                    try:
                        obj.throw(GeneratorExit)
                    except:
                        pass


    def collect_annotations(self) -> tuple[dict[FuncId, FuncAnnotation], dict[FuncId, ModuleVars]]:
        """Collects function type annotations from the observed types."""

        # Finish traces for any generators that may be still running
        for code, per_frame in self.pending_traces.items():
            for tr in per_frame.values():
                if tr.is_generator:
                    self._record_return_type(tr, code, None)


        def most_common_traces(code: CodeType) -> list[CallTrace]:
            """Returns the top X% most common call traces, turning type checking into anomaly detection."""
            counter = self.traces[code]

            threshold = sum(counter.values()) * options.use_top_pct / 100
            cumulative = 0

            traces = list()
            for trace, count in self.traces[code].most_common():
                if cumulative >= threshold:
                    break
                cumulative += count
                traces.append(trace)

            return traces


        def mk_annotation(code: CodeType) -> FuncAnnotation|None:
            func_info = self.functions_visited[code]
            traces = most_common_traces(code)

            parents_arg_types = None
            if func_info.overrides:
                parents_func = func_info.overrides

                if (
                    options.ignore_annotations
                    or not (
                        (parents_arg_types := get_inline_arg_types(parents_func, func_info.args))
                        or (parents_arg_types := get_typeshed_arg_types(parents_func, func_info.args))
                    )
                ):
                    parent_code = parents_func.__code__ if hasattr(parents_func, "__code__") else None
                    if (
                        parent_code
                        and parent_code in self.traces
                        and (ann := mk_annotation(parent_code))
                    ):
                        parents_arg_types = [arg[1] for arg in ann.args]

            if (signature := generalize(traces)) is None:
                logger.info(f"Unable to generalize {func_info.func_id}: inconsistent traces.\n" +
                            f"{[tuple(str(t) for t in s) for s in traces]}")
                return None

            ann = FuncAnnotation(
                args=[
                    (
                        arg.arg_name,
                        merged_types({
                            signature[i],
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
                    (var_name, merged_types(var_types))
                    for var_name, var_types in self.variables[code].items()
                ]
            )

            if logger.level == logging.DEBUG:
                trace_counter = self.traces[code]
                for trace, count in list(trace_counter.items()):
                    logger.debug(
                        "trace " + func_info.func_id.func_name +
                        str(tuple(str(t) for t in trace)) +
                        f" {count}x"
                    )
                logger.debug(
                    "ann   " + func_info.func_id.func_name +
                    str((*(str(arg[1]) for arg in ann.args), str(ann.retval)))
                )
                for var, var_type in list(self.variables[code].items()):
                    logger.debug(
                        "var {func_info.func_id.func_name} {var_type}"
                    )

            return ann

        type_name_adjuster = None
        if options.adjust_type_names:
            type_name_adjuster = AdjustTypeNamesT(self.main_globals)
            self._transform_types(type_name_adjuster)

        class NonSelfCloningT(TypeInfo.Transformer):
            """Clones the given TypeInfo tree, clearing all 'is_self' flags,
               as the type information may not be equivalent to typing.Self in the new context.
            """
            def visit(vself, node: TypeInfo) -> TypeInfo:
                return super().visit(node.replace(is_self=False))

        class CallableT(TypeInfo.Transformer):
            """Updates Callable/Generator/Coroutine type declarations based on observations."""
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                # if 'args' is there, the function is already annotated
                if node.code and (options.ignore_annotations or not node.args) and node.code in self.traces:
                    # TODO we only need the retval, can we avoid computing the entire annotation?
                    if (ann := mk_annotation(node.code)):
                        func_info = self.functions_visited[node.code]
                        # Clone (rather than link to) types from Callable, Generator, etc.,
                        # clearing is_self, as these types may be later replaced with typing.Self.
                        if node.type_obj is abc.Callable:
                            node = node.replace(args=(
                                TypeInfo.list([
                                    NonSelfCloningT().visit(a[1]) for a in ann.args[int(node.is_bound):]
                                ])
                                if not (func_info.varargs or func_info.kwargs) else
                                ...,
                                NonSelfCloningT().visit(ann.retval)
                            ))
                        elif node.type_obj in (abc.Generator, abc.AsyncGenerator):
                            node = NonSelfCloningT().visit(ann.retval)
                        elif node.type_obj is abc.Coroutine:
                            node = node.replace(args=(
                                NoneTypeInfo,
                                NoneTypeInfo,
                                NonSelfCloningT().visit(ann.retval)
                            ))

                return node

        self._transform_types(CallableT())

        class PostponedIteratorArgsT(TypeInfo.Transformer):
            """Replaces a postponed iterator argument evaluation marker with
               its evaluation (performed by CallableT).
            """
            def visit(vself, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                if node.type_obj is PostponedIteratorArg:
                    assert isinstance(node.args[0], TypeInfo)
                    source = node.args[0]
                    if source.args:
                        if source.type_obj is abc.Callable:
                            assert isinstance(source.args[1], TypeInfo)
                            return source.args[1]   # Callable return value
                        else:
                            assert isinstance(source.args[0], TypeInfo)
                            return source.args[0]   # Generator/Iterator yield value
                    return UnknownTypeInfo

                return node

        self._transform_types(PostponedIteratorArgsT())

        if options.use_typing_self:
            self._transform_types(SelfT())

        if not options.use_typing_never:
            self._transform_types(NeverSayNeverT())
        else:
            self._transform_types(NoReturnToNeverT())

        if options.resolve_mocks:
            self._transform_types(ResolveMocksT(type_name_adjuster))

        if options.exclude_test_types:
            self._transform_types(ExcludeTestTypesT())


        finalizers: list[TypeInfo.Transformer] = []

        def finalize(t: TypeInfo) -> TypeInfo:
            for f in finalizers:
                t = f.visit(t)
            return t

        if options.type_depth_limit is not None:
            finalizers.append(DepthLimitT(options.type_depth_limit))

        if options.use_typing_union:
            finalizers.append(TypesUnionT())

        finalizers.append(GeneratorToIteratorT())
        finalizers.append(MakePickleableT())

        non_func_codes = self.variables.keys() - self.traces.keys()

        annotations = {
            self.functions_visited[code].func_id: FuncAnnotation(
                args=[(arg[0], finalize(arg[1])) for arg in annotation.args],
                retval=finalize(annotation.retval),
                varargs=annotation.varargs,
                kwargs=annotation.kwargs,
                variables=[(var[0], finalize(var[1])) for var in annotation.variables]
            )
            for code in self.traces
            if (annotation := mk_annotation(code)) is not None
        }

        module_vars = {
            self.functions_visited[code].func_id: ModuleVars([
                (var_name, finalize(merged_types(var_types)))
                for var_name, var_types in self.variables[code].items()
            ])
            for code in non_func_codes
        }

        return annotations, module_vars


def get_inline_arg_types(
    parents_func: FunctionType|FunctionDescriptor,
    child_args: tuple[ArgInfo, ...]
) -> list[TypeInfo|None] | None:
    """Returns inline type annotations for a parent's method's arguments."""

    if not (co := getattr(parents_func, "__code__", None)):
        return None

    try:
        if not (hints := typing.get_type_hints(parents_func)):
            return None
    except (NameError, TypeError) as e:
        logger.info(f"Error getting type hints for {parents_func} " + 
                    f"({parents_func.__annotations__}): {e}.\n")
        return None

    return (
        # first the positional, looking up by their names given in the parent
        [
            hint2type(hints[arg]) if arg in hints else None
            for arg in co.co_varnames[:co.co_argcount]
        ]
        +
        # then kwonly, going by the order in the child
        [
            hint2type(hints[arg.arg_name]) if arg.arg_name in hints else None
            for arg in child_args[co.co_argcount:]
        ]
    )


def get_typeshed_arg_types(
    parents_func: FunctionDescriptor|FunctionType,
    child_args: tuple[ArgInfo, ...]
) -> list[TypeInfo|None] | None:
    """Returns typeshed type annotations for a parent's method's arguments."""

    def find_def(tree: ast.AST, qualified_name: str) -> list[ast.FunctionDef|ast.AsyncFunctionDef]:
        parts = qualified_name.split('.')
        results = []

        def visit(node, path):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                full_name = '.'.join(path + [node.name])
                if full_name == qualified_name:
                    results.append(node)
            elif isinstance(node, ast.ClassDef):
                new_path = path + [node.name]
                for body_item in node.body:
                    visit(body_item, new_path)
            elif isinstance(node, ast.Module):
                for body_item in node.body:
                    visit(body_item, path)

        visit(tree, [])
        return results

    if stub_ast := typeshed.get_stub_ast(parents_func.__module__):    # FIXME replace __main__?
        if defs := find_def(stub_ast, parents_func.__qualname__):
            #print(ast.dump(defs[0], indent=4))

            # FIXME use eval() in the context of the module and hint2type so
            # as not to have an unqualified string for a "type"

            # first the positional, looking up by their names given in the parent
            pos_args = [
                TypeInfo('', ast.unparse(a.annotation)) if a.annotation else None
                for a in (defs[0].args.posonlyargs + defs[0].args.args)
                if isinstance(a, ast.arg)
            ]

            # then kwonly, going by the order in the child
            kw_args = [
                TypeInfo('', ast.unparse(a.annotation)) if a.annotation else None
                for child_arg_name in child_args[len(pos_args):]
                for a in defs[0].args.kwonlyargs
                if isinstance(a, ast.arg)
                if a.arg == child_arg_name
            ]

            return pos_args + kw_args

    return None


def get_self_type(code, args) -> tuple[TypeInfo|None, TypeInfo|None, FunctionType|FunctionDescriptor|None]:
    if args.args:
        first_arg = args.locals[args.args[0]]

        name = code.co_name
        if (
            name.startswith("__")
            and not name.endswith("__")
            and len(parts := code.co_qualname.split(".")) > 1
        ):
            # parts[-2] may be "<locals>"... that's ok, as we then have
            # a local function and there is no 'Self' to find.
            name = f"_{parts[-2]}{name}"    # private attribute/method

        # if type(first_arg) is type, we may have a @classmethod
        first_arg_class = first_arg if type(first_arg) is type else type(first_arg)

        # @property?
        is_property = isinstance(getattr(type(first_arg), name, None), property)

        # find class that defines that name, in case it's inherited
        defining_class, next_index = next(
            (
                (ancestor, i+1)
                for i, ancestor in enumerate(first_arg_class.__mro__)
                if (
                    (is_property and name in ancestor.__dict__)
                    or (
                        (f := unwrap(ancestor.__dict__.get(name, None)))
                        and getattr(f, "__code__", None) is code
                    )
                )
            ),
            (None, None)
        )

        if not defining_class:
            return None, None, None

        # The first argument is 'Self' and the type of 'Self', in the context of
        # its definition, is "defining_class"; now let's see if this method
        # overrides another
        overrides = None
        if not (
            is_property
            or name in ('__init__', '__new__')  # irrelevant for Liskov
        ):
            overrides = next(
                (
                    # wrapper_descriptor and possibly other native objects may lack __module__
                    f if hasattr(f, "__module__")
                    else FunctionDescriptor(ancestor.__module__, f.__qualname__)
                    for ancestor in first_arg_class.__mro__[next_index:]
                    if (f := unwrap(ancestor.__dict__.get(name, None)))
                    if getattr(f, "__code__", None) is not code
                ),
                None
            )

        return get_type_name(first_arg_class), get_type_name(defining_class), overrides

    return None, None, None


def get_defaults(code, frame) -> dict[str, TypeInfo]:
    if (function := find_function(frame, code)):
        return {
            param_name: get_value_type(param.default)
            for param_name, param in inspect.signature(function).parameters.items()
            if param.default != inspect._empty
        }

    return {}


