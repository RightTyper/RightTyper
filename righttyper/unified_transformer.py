import typing
import builtins
import collections.abc as abc
import types
import libcst as cst
import libcst.matchers as cstm
import re

from righttyper.righttyper_types import (
    ArgumentName,
    Filename,
    FuncInfo,
    FuncAnnotation,
    FunctionName,
    Typename,
)


_BUILTIN_TYPES : frozenset[str] = frozenset({
    t for t in (
        "None",
        *(name for name, value in builtins.__dict__.items() if isinstance(value, type))
    )
})

# FIXME this prevents us from missing out on well-known "typing." types,
# but is risky... change to receiving fully qualified names and simplifying
# them in context.
_TYPING_TYPES : frozenset[str] = frozenset({
    t for t in typing.__all__
})

# Regex for a type hint comment
_TYPE_HINT_COMMENT = re.compile(
    r"#\stype:\s*[^\s]+"
)


# Regex for a function and retval type hint comment
_TYPE_HINT_COMMENT_FUNC = re.compile(
    r"#\stype:\s*\([^\)]*\)\s*->\s*[^\s]+"
)


def _dotted_name_to_nodes(name: str) -> cst.Attribute | cst.Name:
    """Creates Attribute/Name to build a module name, dotted or not."""
    parts = name.split(".")
    if len(parts) == 1:
        return cst.Name(parts[0])
    base: cst.Name | cst.Attribute = cst.Name(parts[0])
    for part in parts[1:]:
        base = cst.Attribute(value=base, attr=cst.Name(part))
    return base

def _nodes_to_dotted_name(node: cst.Attribute|cst.Name|cst.BaseExpression) -> str:
    """Extracts a module name from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        return f"{_nodes_to_dotted_name(node.value)}.{_nodes_to_dotted_name(node.attr)}"

    assert isinstance(node, cst.Name), f"{node=}"
    return node.value

def _nodes_to_top_level_name(node: cst.Attribute|cst.Name|cst.BaseExpression) -> str:
    """Extracts the top-level name (e.g., 'foo' in 'foo.bar') from CST Attribute/Name nodes."""
    while isinstance(node, cst.Attribute):
        node = node.value

    assert isinstance(node, cst.Name), f"{node=}"
    return node.value

def _nodes_to_all_dotted_names(node: cst.Attribute|cst.Name|cst.BaseExpression) -> list[str]:
    """Extracts the list of all module and parent module names from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        names = _nodes_to_all_dotted_names(node.value)
        return [*names, f"{names[-1]}.{node.attr.value}"]

    assert isinstance(node, cst.Name), f"{node=}"
    return [node.value]

def _get_str_attr(obj: object, path: str) -> str|None:
    """Looks for a str-valued attribute along the given dot-separated attribute path."""
    for elem in path.split('.'):
        if obj and isinstance(obj, (list, tuple)):
            obj = obj[0]

        if (obj := getattr(obj, elem, None)) is None:
            break

    return obj if isinstance(obj, str) else None

def _annotation_as_string(annotation: cst.BaseExpression) -> str:
    return cst.Module([cst.SimpleStatementLine([cst.Expr(annotation)])]).code.strip('\n')

def _quote(s: str) -> str:
    s = s.replace('\\', '\\\\')
    return '"' + s.replace('"', '\\"') + '"'


class UnifiedTransformer(cst.CSTTransformer):
    def __init__(
        self,
        filename: str,
        type_annotations: dict[FuncInfo, FuncAnnotation],
        override_annotations: bool,
        module_name: str|None,
        module_names: list[str],
        *,
        use_self: bool = True
    ) -> None:
        self.filename = filename
        self.type_annotations = type_annotations
        self.override_annotations = override_annotations
        self.has_future_annotations = False
        self.module_name = module_name
        self.module_names = sorted(module_names, key=lambda name: -name.count('.'))
        self.change_list: list[tuple[FunctionName, cst.FunctionDef, cst.FunctionDef]] = []
        self.use_self = use_self

    def _module_for(self, name: str) -> tuple[str, str]:
        """Splits a dot name in its module and qualified name parts."""
        # TODO Ideally we'd want to avoid this and just use the type(x).__module__ information
        # we get from the live objects

        for m in self.module_names:
            if name.startswith(m) and (len(name) == len(m) or name[len(m)] == '.'):
                return m, name[len(m)+1:]

        return '', name

    def _is_valid(self, annotation: str) -> bool:
        """Returns whether the annotation can be parsed."""
        # local names such as foo.<locals>.Bar yield this exception
        try:
            cst.parse_expression(annotation)
            return True
        except cst.ParserSyntaxError:
            return False

    def _rename_types(self, annotation: cst.BaseExpression) -> cst.BaseExpression:
        """Renames types in an annotation based on the module name and on any aliases."""

        class Renamer(cst.CSTTransformer):
            def __init__(self, transformer: 'UnifiedTransformer'):
                self.t = transformer

            def visit_Attribute(self, node: cst.Attribute) -> bool:
                return False    # we read the whole name at once, so don't recurse

            def try_replace(self, node: cst.Name|cst.Attribute) -> cst.Name|cst.Attribute:
                name = _nodes_to_dotted_name(node)

                if name in _BUILTIN_TYPES:
                    if name not in self.t.used_names[-1]:
                        return node

                    name = f"builtins.{name}" # somebody overrode a builtin name(!)

                if a := self.t.aliases.get(name):
                    return _dotted_name_to_nodes(a)

                module, rest = self.t._module_for(name)
                if module:
                    if module == self.t.module_name:
                        return _dotted_name_to_nodes(rest)

                    if a := self.t.aliases.get(module):
                        return _dotted_name_to_nodes(f"{a}.{rest}")

                    # does the package name conflict with other definitions?
                    if name.split('.')[0] in self.t.used_names[-1]:
                        alias = "_rt_" + "_".join(module.split("."))
                        self.t.if_checking_aliases[module] = alias
                        return _dotted_name_to_nodes(alias + ("" if rest == "" else f".{rest}"))

                if module == 'builtins':
                    return _dotted_name_to_nodes(name)

                return node

            def leave_Name(self,
                orig_node: cst.Name,
                updated_node: cst.Name
            ) -> cst.Name|cst.Attribute:
                return self.try_replace(updated_node)

            def leave_Attribute(self,
                orig_node: cst.Attribute,
                updated_node: cst.Attribute
            ) -> cst.Name|cst.Attribute:
                return self.try_replace(updated_node)

        return typing.cast(cst.BaseExpression, annotation.visit(Renamer(self)))


    def _unknown_types(self, types: set[str]) -> abc.Iterator[str]:
        """Yields types among those given that are unknown."""
        for t in types:
            if not (
                t.split('.')[0] in self.known_names
                or self._module_for(t)[0] in self.imported_modules
            ):
                yield t


    def visit_Module(self, node: cst.Module) -> bool:
        # Initialize mutable members here, just in case transformer gets reused

        # names used somewhere in a module or class scope
        self.used_names: list[set[str]] = [used_names(node)]

        # currently known global names
        self.known_names: set[str] = set(_BUILTIN_TYPES)

        # global aliases from 'from .. import ..' and 'import .. as ..'
        self.aliases: dict[str, str] = {
            f"typing.{t}": t
            for t in _TYPING_TYPES
        }

        # "import ... as ..." within "if TYPE_CHECKING:".
        # TODO read those in as well so as not to duplicate imports
        self.if_checking_aliases: dict[str, str] = dict()

        # modules imported under their own names
        self.imported_modules: set[str] = set()

        self.unknown_types : set[str] = set()
        self.name_stack : list[str] = []

        self.has_future_annotations = any(
            True for match in cstm.findall(
                node,
                cstm.ImportFrom(
                    module=cstm.Name("__future__"),
                    names=[
                        cstm.ImportAlias(
                            name=cstm.Name("annotations")
                        )
                    ]
                )
            )
        )

        return True


    def visit_If(self, node: cst.If) -> bool:
        if cstm.matches(node, cstm.If(test=cstm.Name("TYPE_CHECKING"))):
            return False    # to ignore imports within TYPE_CHECKING

        return True


    def visit_Import(self, node: cst.Import) -> bool:
        if not self.name_stack: # for now, we only handle global imports
            # node.names could also be cst.ImportStar
            if isinstance(node.names, abc.Sequence):
                for alias in node.names:
                    if alias.asname is not None:
                        self.known_names.add(_nodes_to_top_level_name(alias.asname.name))
                        self.aliases[_nodes_to_dotted_name(alias.name)] = _nodes_to_dotted_name(alias.asname.name)
                    else:
                        self.imported_modules |= set(_nodes_to_all_dotted_names(alias.name))
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        if not self.name_stack: # for now, we only handle global imports
            # node.names could also be cst.ImportStar
            if isinstance(node.names, abc.Sequence):
                for alias in node.names:
                    self.known_names.add(
                        _nodes_to_top_level_name(
                            alias.asname.name if alias.asname is not None else alias.name
                        )
                    )

                    source: list[str] = []

                    if node.relative:
                        if self.module_name is None:
                            # TODO warn: cannot handle import: module name unknown
                            return False

                        source = self.module_name.split('.')[:-len(node.relative)]
                        if not source:
                            # TODO warn: invalid import (goes out of package space)
                            return False

                    source += _nodes_to_dotted_name(node.module).split('.') if node.module else []

                    self.aliases[f"{'.'.join(source)}.{_nodes_to_dotted_name(alias.name)}"] = \
                        _nodes_to_dotted_name(
                            alias.asname.name if alias.asname is not None else alias.name
                        )

        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        name_source = list_rindex(self.name_stack, '<locals>') # neg. index of last function, or 0 (for globals)
        self.name_stack.append(node.name.value)
        self.used_names.append(self.used_names[name_source] | used_names(node))
        return True

    def leave_ClassDef(self, orig_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        # a class is known once its definition is done
        self.known_names.add(".".join(self.name_stack))
        self.name_stack.pop()
        self.used_names.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name_source = list_rindex(self.name_stack, '<locals>') # neg. index of last function, or 0 (for globals)
        self.name_stack.extend([node.name.value, "<locals>"])
        self.used_names.append(self.used_names[name_source] | used_names(node))
        return True

    def _try_rename_to_self(self, annotation: Typename) -> Typename:
        if self.use_self and self.module_name:
            # TODO this doesn't handle composite names, such as "list[Self]"... do we care?
            if annotation == '.'.join((self.module_name, *self.name_stack)):
                return Typename("typing.Self")
        return annotation

    def _process_parameter(self, parameter: cst.Param, ann: FuncAnnotation) -> cst.Param:
        """Processes a parameter, either returning an updated parameter or the original one."""
        if (
            not (parameter.annotation is None or self.override_annotations)
            or (annotation := next(
                (
                    annotation for arg, annotation in ann.args
                    if arg == parameter.name.value
                ), None)
            ) is None
        ):
            return parameter

        annotation = self._try_rename_to_self(annotation)

        if not self._is_valid(annotation):
            return parameter

        annotation_expr: cst.BaseExpression = cst.parse_expression(annotation)
        annotation_expr = self._rename_types(annotation_expr)
        unknown_types = set(self._unknown_types(types_in_annotation(annotation_expr)))
        self.unknown_types |= unknown_types

        if not self.has_future_annotations and (unknown_types - _TYPING_TYPES):
            annotation_expr = cst.SimpleString(_quote(_annotation_as_string(annotation_expr)))

        new_par = parameter.with_changes(
            annotation=cst.Annotation(annotation=annotation_expr)
        )

        # remove per-parameter type hint comment for non-last parameter
        if ((comment := _get_str_attr(new_par, "comma.whitespace_after.first_line.comment.value"))
            and _TYPE_HINT_COMMENT.match(comment)):
            new_par = new_par.with_changes(
                comma=new_par.comma.with_changes(   # type: ignore[union-attr]
                    whitespace_after=new_par.comma.whitespace_after.with_changes( # type: ignore[union-attr]
                        first_line=cst.TrailingWhitespace()
                    )
                )
            )

        # remove per-parameter type hint comment for last parameter
        if ((comment := _get_str_attr(new_par, "whitespace_after_param.first_line.comment.value"))
            and _TYPE_HINT_COMMENT.match(comment)):
            new_par = new_par.with_changes(
                whitespace_after_param=new_par.whitespace_after_param.with_changes(
                    first_line=cst.TrailingWhitespace()
                )
            )

        return new_par

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        name = ".".join(self.name_stack[:-1])
        self.name_stack.pop()
        self.name_stack.pop()
        self.used_names.pop()
        key = FuncInfo(Filename(self.filename), FunctionName(name))

        if ann := self.type_annotations.get(key):
            for attr_name in ['params', 'kwonly_params', 'posonly_params']:
                if getattr(updated_node.params, attr_name):
                    new_parameters = []
                    for parameter in getattr(updated_node.params, attr_name):
                        new_parameters.append(self._process_parameter(parameter, ann))

                    updated_node = updated_node.with_changes(
                        params=updated_node.params.with_changes(
                            **{attr_name: new_parameters}
                        )
                    )

            for attr_name in ['star_arg', 'star_kwarg']:
                attr = getattr(updated_node.params, attr_name)
                if isinstance(attr, cst.Param):
                    updated_node = updated_node.with_changes(
                        params=updated_node.params.with_changes(
                            **{attr_name: self._process_parameter(attr, ann)}
                        )
                    )

            if ((updated_node.returns is None or self.override_annotations)
                and ann.retval is not None
            ):
                annotation = self._try_rename_to_self(ann.retval)
                if self._is_valid(annotation):
                    annotation_expr = cst.parse_expression(annotation)
                    annotation_expr = self._rename_types(annotation_expr)
                    unknown_types = set(self._unknown_types(types_in_annotation(annotation_expr)))
                    self.unknown_types |= unknown_types

                    if not self.has_future_annotations and (unknown_types - _TYPING_TYPES):
                        annotation_expr = cst.SimpleString(_quote(_annotation_as_string(annotation_expr)))

                    updated_node = updated_node.with_changes(
                        returns=cst.Annotation(annotation=annotation_expr),
                    )

                    # remove "(...) -> retval"-style type hint comment
                    if ((comment := _get_str_attr(updated_node, "body.body.leading_lines.comment.value"))
                        and _TYPE_HINT_COMMENT_FUNC.match(comment)):
                        updated_node = updated_node.with_changes(
                            body=updated_node.body.with_changes(
                                body=(updated_node.body.body[0].with_changes(leading_lines=[]),
                                      *updated_node.body.body[1:])
                            )
                        )

            # remove single-line type hint comment in the same line as the 'def'
            if ((comment := _get_str_attr(updated_node, "body.header.comment.value"))
                and _TYPE_HINT_COMMENT_FUNC.match(comment)):
                updated_node = updated_node.with_changes(
                    body=updated_node.body.with_changes(
                        header=cst.TrailingWhitespace()))

            self.change_list.append((key.func_name, original_node, updated_node))

        return updated_node

    # ConstructImportTransformer logic
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        # Collect `from __future__` imports and remove them
        future_imports: list[cst.BaseStatement] = []
        new_body: list[cst.BaseStatement] = []
        stmt: cst.BaseStatement

        from_future = cstm.SimpleStatementLine(
            body=[cstm.ImportFrom(
                module=cstm.Name("__future__")
            )]
        )

        for stmt in updated_node.body:
            if cstm.matches(stmt, from_future):
                # Collect future imports to lift later
                future_imports.append(stmt)
            else:
                new_body.append(stmt)

        missing_modules = {
            mod
            for mod in (
                self._module_for(t)[0]
                for t in self.unknown_types - _TYPING_TYPES
                if '.' in t
                and t.split('.')[0] not in self.known_names
            )
            if mod != ''
        }

        def stmt_index(body: list[cst.BaseStatement], pattern: cstm.BaseMatcherNode) -> int|None:
            for i, stmt in enumerate(body):
                if cstm.matches(stmt, pattern):
                    return i

            return None

        def find_beginning(body: list[cst.BaseStatement]) -> int:
            for i, stmt in enumerate(body):
                if not cstm.matches(stmt,
                    cstm.OneOf(
                        cstm.EmptyLine(),
                        cstm.SimpleStatementLine(
                            body=[cstm.Expr(cstm.SimpleString())]
                        )
                    )
                ):
                    break

            return i

        if_type_checking_position = stmt_index(new_body, cstm.If(
                test=cstm.Name('TYPE_CHECKING'),
                body=cstm.IndentedBlock()
            )
        )

        # Add additional type checking imports if needed
        if missing_modules or self.if_checking_aliases:
            existing_body = [*(typing.cast(cst.If, new_body[if_type_checking_position]).body.body
                               if if_type_checking_position is not None
                               else ())]

            # TODO delete modules already imported

            new_stmt: cst.BaseStatement = cst.If(
                test=cst.Name("TYPE_CHECKING"),
                body=cst.IndentedBlock(
                    body=existing_body + [
                        cst.SimpleStatementLine([
                            cst.Import([cst.ImportAlias(_dotted_name_to_nodes(m))])
                        ])
                        for m in sorted(missing_modules)
                    ] + [
                        cst.SimpleStatementLine([
                            cst.Import([cst.ImportAlias(
                                name=_dotted_name_to_nodes(m),
                                asname=cst.AsName(cst.Name(a))
                            )])
                        ])
                        for m, a in sorted(self.if_checking_aliases.items())
                    ]
                )
            )

            if if_type_checking_position is not None:
                new_body[if_type_checking_position] = new_stmt
            else:
                if_type_checking_position = find_beginning(new_body)
                new_body.insert(if_type_checking_position, new_stmt)

            if 'TYPE_CHECKING' not in self.known_names:
                self.unknown_types.add('TYPE_CHECKING')

        # Emit "from typing import ..."
        if (typing_types := (self.unknown_types & _TYPING_TYPES)):
            existing = stmt_index(new_body, cstm.SimpleStatementLine(
                    body=[cstm.ImportFrom(module=cstm.Name("typing"))]
                )
            )

            if existing is not None:
                for alias in cstm.findall(new_body[existing], cstm.ImportAlias()):
                    typing_types.add(_nodes_to_dotted_name(typing.cast(cst.ImportAlias, alias).name))

            new_stmt = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name(value="typing"),
                        names = [
                            cst.ImportAlias(name=_dotted_name_to_nodes(t))
                            # sort 'TYPE_CHECKING' first just because it seems neater that way
                            for t in sorted(typing_types, key=lambda x: (x != 'TYPE_CHECKING', x))
                        ]
                    )
                ]
            )

            if (existing is not None
                and (if_type_checking_position is None
                     or existing < if_type_checking_position)
            ):
                new_body[existing] = new_stmt
            else: 
                position = if_type_checking_position if if_type_checking_position is not None \
                           else find_beginning(new_body)
                new_body.insert(position, new_stmt)


        b = find_beginning(new_body)
        new_body[b:b] = future_imports

        return updated_node.with_changes(body=new_body)


    def get_signature_changes(self: typing.Self) -> list[tuple[FunctionName, str, str]]:
        return [
            (name, old_sig, new_sig)
            for name, old_sig, new_sig in (
                (name, format_signature(old), format_signature(new))
                for name, old, new in self.change_list
            )
            if new_sig != old_sig
        ]


def types_in_annotation(annotation: cst.BaseExpression) -> set[str]:
    """Extracts all type names included in a type annotation."""

    class TypeNameExtractor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit_Name(self, node: cst.Name) -> bool:
            self.names.add(node.value)
            return False 

        def visit_Attribute(self, node: cst.Attribute) -> bool:
            self.names.add(_nodes_to_dotted_name(node))
            return False 

    extractor = TypeNameExtractor()
    annotation.visit(extractor)
    return extractor.names


def used_names(node: cst.Module|cst.ClassDef|cst.FunctionDef) -> set[str]:
    """Extracts the names in a module or class."""

    # FIXME handle 'global' and 'nonlocal'

    names: set[str] = set()

    class Extractor(cst.CSTVisitor):
        def __init__(self):
            self.in_scope = False

        def visit_Module(self, node: cst.Module) -> bool:
            self.in_scope = True
            return True

        def visit_ClassDef(self, node: cst.ClassDef) -> bool:
            if self.in_scope:
                names.add(node.name.value)
                return False

            self.in_scope = True
            return True

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            if self.in_scope:
                names.add(node.name.value)
                return False

            self.in_scope = True
            return True

        def visit_Assign(self, node: cst.Assign) -> bool:
            for t in node.targets:
                if isinstance(t.target, cst.Name):
                    names.add(t.target.value)
                elif isinstance(t.target, cst.Tuple):
                    for el in t.target.elements:
                        if isinstance(el.value, cst.Name):
                            names.add(el.value.value)
            return True

        def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
            if isinstance(node.target, cst.Name):
                names.add(node.target.value)
            return True

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
            # node.names could also be cst.ImportStar
            if isinstance(node.names, abc.Sequence):
                for alias in node.names:
                    if not alias.asname:
                        names.add(_nodes_to_top_level_name(alias.name))
            return True

        def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
            names.add(_nodes_to_top_level_name(node.target))
            return True

        def visit_AsName(self, node: cst.AsName) -> bool:
            if isinstance(node.name, (cst.Tuple, cst.List)):
                for el in node.name.elements:
                    names.add(_nodes_to_top_level_name(el.value))
            else:
                names.add(_nodes_to_top_level_name(node.name))

            return True

    node.visit(Extractor())
    return names


def list_rindex(l: list, item: object) -> int:
    """Returns either a negative index for the last occurrence of 'item' on the list,
       or 0 if not found."""
    try:
        return -1 - l[::-1].index(item)
    except ValueError:
        return 0


def format_signature(f: cst.FunctionDef) -> str:
    """Formats the signature of a function."""

    class BodyRemover(cst.CSTTransformer):
        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.FunctionDef:
            return updated_node.with_changes(body=cst.IndentedBlock(body=[
                cst.SimpleStatementLine(body=[])
            ]))

        def leave_Decorator(
            self, original_node: cst.Decorator, updated_node: cst.Decorator
        ) -> cst.RemovalSentinel:
            return cst.RemoveFromParent()

    bodyless = typing.cast(cst.FunctionDef, f.visit(BodyRemover()))
    sig = cst.Module([bodyless]).code.strip()

    # It's easier to let libcst generate "pass" for an empty body and then remove it
    # than to find a way to have it emit a bodyless function...
    if sig.endswith("pass"):
        sig = sig[:-4].strip()

    return sig
