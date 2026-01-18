import ast
import types
from dataclasses import dataclass, field, replace
import collections.abc as abc


@dataclass
class CodeVars:
    """Identifies variables and attributes for a code object, facilitating their capture."""

    # qualified name of code object in whose scope we're storing variables
    scope: str

    # that code object
    scope_code: types.CodeType | None = None

    # maps name in f_locals to name in scope
    variables: dict[str, str] = field(default_factory=dict)

    # qualified name of this code object's class, if it's a method
    class_name: str | None = None

    # unique object that identifies the class
    class_key: object | None = None

    # name of 'self' for this object, if any
    self: str | None = None

    # the 'self' attributes
    attributes: set[str]|None = None

    # maps variable name to its initial constant's type (if first assignment is a constant)
    initial_constants: dict[str, type] = field(default_factory=dict)

    # maps attribute name to its initial constant's type (for self.x = None etc.)
    attribute_initial_constants: dict[str, type] = field(default_factory=dict)


"""Maps code objects to the variables assigned/bound within each object."""
code2variables: dict[types.CodeType, CodeVars] = dict()


def _extract_name(node: ast.AST) -> str:
    parts: list[str] = []

    if isinstance(node, ast.Subscript):
        node = node.value

    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        parts.append(node.id)

    return ".".join(reversed(parts))


def _iter_target_atoms(target: ast.AST) -> abc.Iterator[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_target_atoms(elt)
    elif isinstance(target, ast.Starred):
        yield from _iter_target_atoms(target.value)
    else:
        yield target


def _from_pattern(p: ast.pattern) -> abc.Iterator[str]:
    if isinstance(p, ast.MatchAs):
        if p.name: yield p.name
        if p.pattern: yield from _from_pattern(p.pattern)
    elif isinstance(p, ast.MatchStar):
        if p.name: yield p.name
    elif isinstance(p, ast.MatchMapping):
        for sub in p.patterns:
            yield from _from_pattern(sub)
        if p.rest: yield p.rest
    elif isinstance(p, ast.MatchSequence):
        for sub in p.patterns:
            yield from _from_pattern(sub)
    elif isinstance(p, ast.MatchClass):
        for sub in p.patterns:
            yield from _from_pattern(sub)
        for sub in p.kwd_patterns:
            yield from _from_pattern(sub)
    elif isinstance(p, ast.MatchOr):
        # Python enforces that the left side of "or" must match the right
        yield from _from_pattern(p.patterns[0])


@dataclass
class ClassInfo:
    name: str
    attributes: set[str] = field(default_factory=set)
    attribute_initial_constants: dict[str, type] = field(default_factory=dict)


class VariableFinder(ast.NodeVisitor):
    """
    Walks the AST, collecting all variable names (dot-separated where applicable) that are
    assigned or bound within each scope.
    Imports, comprehensions, generator expressions are ignored.
    """
    def __init__(self) -> None:
        # Used to build an item's qualified name
        self._qualname_stack: list[str] = []

        # Holds the qualified name of the code object assigning to the variable
        self._code_stack: list[str] = ['<module>']

        # _qualname_stack at the beginning of the current function; used to determine
        # where and under which name we store variables
        self._scope_stack: list[list[str]] = [[]]

        # Qualified name of the current class
        self._class_stack: list[ClassInfo] = []

        # Holds the current name of 'self', or None if none
        self._self_stack: list[str|None] = [None]

        # Holds the set of names that aren't local variables;
        # currently only includes arguments
        self._not_locals_stack: list[set[str]] = [set()]

        # Resulting map of executing code object to their CodeVars
        self.code_vars: dict[str, CodeVars] = dict()

    def _record_variable(self, name: str, value: ast.expr | None = None) -> None:
        if name in self._not_locals_stack[-1]:
            return

        scope = self._scope_stack[-1]
        codevars = self.code_vars.setdefault(self._code_stack[-1],
            CodeVars('.'.join(scope[:-1]) if scope else '<module>') # -1 to omit "<locals>"
        )

        # Only record if this is the first assignment (variable definition)
        if name not in codevars.variables:
            codevars.variables[name] = '.'.join(self._qualname_stack[len(scope):] + [name])

            # If initial value is a constant, record its type
            if value is not None and isinstance(value, ast.Constant):
                codevars.initial_constants[name] = type(value.value)

    def _record_target(self, t: ast.AST, value: ast.expr | None = None) -> None:
        if isinstance(t, ast.Name):
            self._record_variable(t.id, value)
            if t.id == self._self_stack[-1]:
                self._self_stack[-1] = None # a new assignment masked 'self'
        elif (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)):
            if t.value.id == self._self_stack[-1]:
                class_info = self._class_stack[-1]
                class_info.attributes.add(t.attr)
                # Capture initial constant for attribute (only first assignment)
                if t.attr not in class_info.attribute_initial_constants:
                    if value is not None and isinstance(value, ast.Constant):
                        class_info.attribute_initial_constants[t.attr] = type(value.value)

    def _qualname(self) -> str:
        return '.'.join(self._qualname_stack)

    def visit_FunctionDef(self, node: ast.FunctionDef|ast.AsyncFunctionDef) -> None:
        decorator_names = [
            n.id
            for n in node.decorator_list
            if isinstance(n, ast.Name)
        ]

        is_method = (
            bool(self._qualname_stack)
            and self._qualname_stack[-1] != '<locals>'
            and 'staticmethod' not in decorator_names
            and 'classmethod' not in decorator_names
        )

        arguments = [
            n.arg
            for n in ast.walk(node.args)
            if isinstance(n, ast.arg)
        ]

        self_visible = (
            bool(self._self_stack[-1])
            and not any(
                # 'self' masked by an inner function's argument
                arg == self._self_stack[-1]
                for arg in arguments
            )
        )

        self._qualname_stack.append(node.name)
        self._code_stack.append(self._qualname())
        self._qualname_stack.append('<locals>')
        self._scope_stack.append([*self._qualname_stack])
        if is_method:
            self._self_stack.append(arguments[0] if arguments else None)
        elif self_visible:
            self._self_stack.append(self._self_stack[-1])
        else:
            self._self_stack.append(None)
        self._not_locals_stack.append(set(arguments))

        if self._class_stack:
            class_info = self._class_stack[-1]
            self.code_vars[self._code_stack[-1]] = CodeVars(
                '.'.join(self._qualname_stack[:-1]),    # -1 to omit "<locals>"
                class_name=class_info.name,
                self=self._self_stack[-1],
                attributes=class_info.attributes if self._self_stack[-1] else None,
                attribute_initial_constants=class_info.attribute_initial_constants if self._self_stack[-1] else {}
            )

        self.generic_visit(node)

        self._not_locals_stack.pop()
        self._self_stack.pop()
        self._qualname_stack.pop()
        self._code_stack.pop()
        self._scope_stack.pop()
        self._qualname_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_variable(node.name)
        self._qualname_stack.append(node.name)
        self._code_stack.append(self._qualname())
        self._class_stack.append(ClassInfo(self._code_stack[-1]))

        self.generic_visit(node)

        self._class_stack.pop()
        self._code_stack.pop()
        self._qualname_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # any NamedExpr within stay within the lambda's scope
        pass

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only pass value for simple single-target assignments like "x = None"
        # or "self.x = None". We skip multi-target (x = y = 1) and tuple
        # unpacking (a, b = 0, 1) to keep the implementation simple.
        value = node.value if (len(node.targets) == 1
                               and isinstance(node.targets[0], (ast.Name, ast.Attribute))) else None
        for tgt in node.targets:
            for t in _iter_target_atoms(tgt):
                self._record_target(t, value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # require value to filter out pure type declarations
        if node.value is not None and node.target is not None:
            for t in _iter_target_atoms(node.target):
                self._record_target(t, node.value)
        self.generic_visit(node)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        # Is 'type X = ...' a declaration or variable definition?
        # For now, we err on the side of capturing it.
        for t in _iter_target_atoms(node.name):
            self._record_target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for t in _iter_target_atoms(item.optional_vars):
                    self._record_target(t)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for t in _iter_target_atoms(item.optional_vars):
                    self._record_target(t)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if isinstance(node.name, str):
            self._record_variable(node.name)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        # Bind names from patterns
        for case in node.cases:
            for nm in _from_pattern(case.pattern):
                self._record_variable(nm)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        pass # don't need these

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        pass # don't need these


def _walk_code_objects(co: types.CodeType) -> abc.Iterator[types.CodeType]:
    """Iterates through all code objects within a code object."""
    yield co

    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from _walk_code_objects(c)


def map_variables(tree: ast.Module, module_code: types.CodeType) -> dict[types.CodeType, CodeVars]:
    """Creates a map of code objects to the variables assigned to in that code,
       to facilitate variable sampling."""

    f = VariableFinder()
    f.visit(tree)

    qualname2code: dict[str|None, types.CodeType] = {
        co.co_qualname: co
        for co in _walk_code_objects(module_code)
    }

    return {
        co: replace(
            codevars,
            scope_code=scope_code,
            class_key=qualname2code.get(codevars.class_name)
        )
        for co in qualname2code.values()
        if (codevars := f.code_vars.get(co.co_qualname))
        if (scope_code := qualname2code.get(codevars.scope))
    }
