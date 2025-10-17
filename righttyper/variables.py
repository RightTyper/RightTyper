import ast
import types
from dataclasses import dataclass
from collections import defaultdict
import collections.abc as abc


@dataclass(eq=True, frozen=True)
class CodeVars:
    scope: types.CodeType | None    # code object in whose scope we're storing variables
    variables: dict[str, str]       # maps name in scope to name in f_locals


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


class VariableFinder(ast.NodeVisitor):
    """
    Walks the AST, collecting all variable names (dot-separated where applicable) that are
    assigned or bound within each scope.
    Imports, comprehensions, generator expressions are ignored.
    """
    def __init__(self) -> None:
        self._qualname_stack: list[str] = []
        self._code_stack: list[str] = ['<module>']
        self._scope_stack: list[list[str]] = [[]]
        self.code_vars: dict[str, tuple[str, dict[str, str]]] = dict()

    def _record_name(self, name: str) -> None:
        scope = self._scope_stack[-1]
        codevars = self.code_vars.setdefault(self._code_stack[-1],
            # -1 to omit "<locals>"
            ('.'.join(scope[:-1]) if scope else '<module>', {})
        )
        dst_name = '.'.join(self._qualname_stack[len(scope):] + [name])
        codevars[1][dst_name] = name

    def _record_target(self, t: ast.AST) -> None:
        self._record_name(_extract_name(t))

    def _qualname(self) -> str:
        return '.'.join(self._qualname_stack)

    def visit_FunctionDef(self, node: ast.FunctionDef|ast.AsyncFunctionDef) -> None:
        self._qualname_stack.append(node.name)
        self._code_stack.append(self._qualname())
        self._qualname_stack.append('<locals>')
        self._scope_stack.append([*self._qualname_stack])
        self.generic_visit(node)
        self._qualname_stack.pop();
        self._code_stack.pop()
        self._scope_stack.pop()
        self._qualname_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_name(node.name)
        self._qualname_stack.append(node.name)
        self._code_stack.append(self._qualname())
        self.generic_visit(node)
        self._code_stack.pop()
        self._qualname_stack.pop();

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # any NamedExpr within stay within the lambda's scope
        pass

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            for t in _iter_target_atoms(tgt):
                self._record_target(t)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # require value to filter out pure type declarations
        if node.value is not None and node.target is not None:
            for t in _iter_target_atoms(node.target):
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
            self._record_name(node.name)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        # Bind names from patterns
        for case in node.cases:
            for nm in _from_pattern(case.pattern):
                self._record_name(nm)
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


def map_variables(tree: ast.Module, module_code: types.CodeType) -> dict[types.CodeType, set[str]]:
    """Creates a map of code objects to the variables assigned to in that code,
       to facilitate variable sampling."""

    f = VariableFinder()
    f.visit(tree)

    qualname2code = {
        co.co_qualname: co
        for co in _walk_code_objects(module_code)
    }

    return {
        co: CodeVars(qualname2code.get(code_vars[0]), code_vars[1])
        for co in qualname2code.values()
        if (code_vars := f.code_vars.get(co.co_qualname))
    }
