import ast
import types
from dataclasses import dataclass, field, replace
import collections.abc as abc


# Operator/builtin → dunder method maps for desugaring
_BINOP_TO_DUNDER: dict[type, str] = {
    ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__", ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__", ast.Mod: "__mod__", ast.Pow: "__pow__",
    ast.LShift: "__lshift__", ast.RShift: "__rshift__",
    ast.BitOr: "__or__", ast.BitXor: "__xor__", ast.BitAnd: "__and__",
    ast.MatMult: "__matmul__",
}

_AUGASSIGN_TO_DUNDER: dict[type, str] = {
    ast.Add: "__iadd__", ast.Sub: "__isub__", ast.Mult: "__imul__", ast.Div: "__itruediv__",
    ast.FloorDiv: "__ifloordiv__", ast.Mod: "__imod__", ast.Pow: "__ipow__",
    ast.LShift: "__ilshift__", ast.RShift: "__irshift__",
    ast.BitOr: "__ior__", ast.BitXor: "__ixor__", ast.BitAnd: "__iand__",
    ast.MatMult: "__imatmul__",
}

_UNARYOP_TO_DUNDER: dict[type, str] = {
    ast.UAdd: "__pos__", ast.USub: "__neg__", ast.Invert: "__invert__",
    # ast.Not uses __bool__, but everything has that
}

_CMPOP_TO_DUNDER: dict[type, str] = {
    ast.Eq: "__eq__", ast.NotEq: "__ne__",
    ast.Lt: "__lt__", ast.LtE: "__le__", ast.Gt: "__gt__", ast.GtE: "__ge__",
}

_BUILTIN_TO_DUNDER: dict[str, str] = {
    "len": "__len__", "iter": "__iter__", "next": "__next__",
    "hash": "__hash__", "abs": "__abs__", "repr": "__repr__",
    "reversed": "__reversed__", "str": "__str__", "bool": "__bool__",
    "int": "__int__", "float": "__float__", "round": "__round__",
    "complex": "__complex__", "bytes": "__bytes__",
}


@dataclass
class CodeVars:
    """Identifies variables and attributes for a code object, facilitating their capture."""

    # qualified name of code object in whose scope we're storing variables
    scope: str

    # that code object
    scope_code: types.CodeType | None = None

    # prefix for qualified variable names (e.g., "C." for class C, "" for functions)
    var_prefix: str = ""

    # local variable names: var_name -> initial constant type (or None if not a constant)
    variables: dict[str, type | None] = field(default_factory=dict)

    # qualified name of this code object's class, if it's a method
    class_name: str | None = None

    # unique object that identifies the class
    class_key: object | None = None

    # name of 'self' for this object, if any
    self: str | None = None

    # name of 'cls' for this object (classmethods), if any
    cls: str | None = None

    # the 'self' attributes: attr_name -> initial constant type (or None if not a constant)
    attributes: dict[str, type | None] | None = None

    # class-level attributes (cls.x or class body assignments): attr_name -> initial constant type
    class_attributes: dict[str, type | None] | None = None

    # attributes accessed on each variable (from static analysis of the function body)
    accessed_attributes: dict[str, set[str]] = field(default_factory=dict)


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
    # Instance attributes (self.x): attr_name -> initial constant type (or None if not a constant)
    attributes: dict[str, type | None] = field(default_factory=dict)
    # Class-level attributes (cls.x or class body assignments): attr_name -> initial constant type
    class_attributes: dict[str, type | None] = field(default_factory=dict)


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

        # Holds the current name of 'cls' (for classmethods), or None if none
        self._cls_stack: list[str|None] = [None]

        # Holds the set of names that aren't local variables;
        # currently only includes arguments
        self._not_locals_stack: list[set[str]] = [set()]

        # Resulting map of executing code object to their CodeVars
        self.code_vars: dict[str, CodeVars] = dict()

        # Alias tracking: variable -> set of variables it could be (for attribute propagation).
        # Stacked per function scope so siblings don't see each other's aliases; nested
        # functions inherit a copy of their parent scope's aliases (closure refs preserved).
        self._aliases_stack: list[dict[str, set[str]]] = [{}]

        # Attribute accesses collected for the current scope
        # Stored as code_qualname -> {var_name -> {attr_names}}
        self._accessed_attributes: dict[str, dict[str, set[str]]] = {}

    def _record_variable(self, name: str, value: ast.expr | None = None) -> None:
        if name in self._not_locals_stack[-1]:
            return

        scope = self._scope_stack[-1]
        codevars = self.code_vars.setdefault(self._code_stack[-1],
            CodeVars('.'.join(scope[:-1]) if scope else '<module>') # -1 to omit "<locals>"
        )

        # Compute var_prefix once (for class bodies, this is "C." or "C.D.", for functions it's "")
        if not codevars.var_prefix:
            prefix_parts = self._qualname_stack[len(scope):]
            if prefix_parts:
                codevars.var_prefix = '.'.join(prefix_parts) + "."

        # Only record if this is the first assignment (variable definition)
        if name not in codevars.variables:
            const_type = type(value.value) if (value is not None and isinstance(value, ast.Constant)) else None
            codevars.variables[name] = const_type

    def _record_alias(self, name: str, value: ast.expr | None) -> None:
        """Track name-to-name aliases for attribute access propagation.
        Unions aliases across branches (conservative: over-approximates)."""
        aliases = self._aliases_stack[-1]
        if value is not None and isinstance(value, ast.Name):
            # y = x → y aliases whatever x aliases (plus x itself), unioned with any prior aliases
            new_aliases = aliases.get(value.id, set()) | {value.id}
            aliases[name] = aliases.get(name, set()) | new_aliases
        elif name in aliases:
            # y = <non-name> → y is no longer a pure alias, but keep prior aliases
            # (could be from a different branch)
            pass

    def _resolve_aliases(self, name: str) -> set[str]:
        """Resolve a variable name to all names it could refer to (including itself)."""
        return self._aliases_stack[-1].get(name, set()) | {name}

    def _record_attribute_access(self, var_name: str, attr_name: str) -> None:
        """Record that attr_name is accessed on var_name (and its aliases)."""
        code_name = self._code_stack[-1]
        attrs = self._accessed_attributes.setdefault(code_name, {})
        for name in self._resolve_aliases(var_name):
            attrs.setdefault(name, set()).add(attr_name)

    def _record_target(self, t: ast.AST, value: ast.expr | None = None) -> None:
        if isinstance(t, ast.Name):
            self._record_variable(t.id, value)
            self._record_alias(t.id, value)
            if t.id == self._self_stack[-1]:
                self._self_stack[-1] = None # a new assignment masked 'self'
            if t.id == self._cls_stack[-1]:
                self._cls_stack[-1] = None # a new assignment masked 'cls'
        elif (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)):
            if t.value.id == self._self_stack[-1]:
                class_info = self._class_stack[-1]
                # Only record first assignment; capture initial constant type if applicable
                if t.attr not in class_info.attributes:
                    const_type = type(value.value) if (value is not None and isinstance(value, ast.Constant)) else None
                    class_info.attributes[t.attr] = const_type
            # cls.x tracking for classmethods
            elif self._cls_stack[-1] and t.value.id == self._cls_stack[-1]:
                class_info = self._class_stack[-1]
                # Only record first assignment; capture initial constant type if applicable
                if t.attr not in class_info.class_attributes:
                    const_type = type(value.value) if (value is not None and isinstance(value, ast.Constant)) else None
                    class_info.class_attributes[t.attr] = const_type

    def _qualname(self) -> str:
        return '.'.join(self._qualname_stack)

    def visit_FunctionDef(self, node: ast.FunctionDef|ast.AsyncFunctionDef) -> None:
        decorator_names = [
            n.id
            for n in node.decorator_list
            if isinstance(n, ast.Name)
        ]

        is_in_class = (
            bool(self._qualname_stack)
            and self._qualname_stack[-1] != '<locals>'
        )
        is_staticmethod = 'staticmethod' in decorator_names
        is_classmethod = 'classmethod' in decorator_names
        is_method = is_in_class and not is_staticmethod and not is_classmethod

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

        cls_visible = (
            bool(self._cls_stack[-1])
            and not any(
                # 'cls' masked by an inner function's argument
                arg == self._cls_stack[-1]
                for arg in arguments
            )
        )

        self._qualname_stack.append(node.name)
        self._code_stack.append(self._qualname())
        self._qualname_stack.append('<locals>')
        self._scope_stack.append([*self._qualname_stack])

        # Handle self/cls stack for different method types
        if is_method:
            self._self_stack.append(arguments[0] if arguments else None)
            self._cls_stack.append(None)
        elif is_classmethod:
            self._self_stack.append(None)
            self._cls_stack.append(arguments[0] if arguments else None)
        elif self_visible:
            self._self_stack.append(self._self_stack[-1])
            self._cls_stack.append(self._cls_stack[-1] if cls_visible else None)
        elif cls_visible:
            self._self_stack.append(None)
            self._cls_stack.append(self._cls_stack[-1])
        else:
            self._self_stack.append(None)
            self._cls_stack.append(None)

        self._not_locals_stack.append(set(arguments))
        # Inherit a copy of the parent's aliases so closure refs still propagate,
        # but isolate this scope's new aliases from siblings.
        self._aliases_stack.append(dict(self._aliases_stack[-1]))

        if self._class_stack:
            class_info = self._class_stack[-1]
            self.code_vars[self._code_stack[-1]] = CodeVars(
                '.'.join(self._qualname_stack[:-1]),    # -1 to omit "<locals>"
                class_name=class_info.name,
                self=self._self_stack[-1],
                cls=self._cls_stack[-1],
                attributes=class_info.attributes if self._self_stack[-1] else None,
                class_attributes=class_info.class_attributes,
            )

        self.generic_visit(node)

        self._aliases_stack.pop()
        self._not_locals_stack.pop()
        self._self_stack.pop()
        self._cls_stack.pop()
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
        class_info = ClassInfo(self._code_stack[-1])
        self._class_stack.append(class_info)

        self.generic_visit(node)

        # Update existing CodeVars (created during class body processing) with class_attributes,
        # or create one if it doesn't exist yet
        class_qualname = self._code_stack[-1]
        if class_qualname in self.code_vars:
            existing = self.code_vars[class_qualname]
            existing.class_name = class_info.name
            existing.class_attributes = class_info.class_attributes
        else:
            self.code_vars[class_qualname] = CodeVars(
                class_qualname,  # scope is the class itself
                class_name=class_info.name,
                class_attributes=class_info.class_attributes,
            )

        self._class_stack.pop()
        self._code_stack.pop()
        self._qualname_stack.pop()

    def _is_builtin(self, name: str) -> bool:
        """Check if name refers to a builtin (not shadowed by a local or parameter)."""
        return (name not in self._not_locals_stack[-1]  # not a parameter
                and name not in self.code_vars.get(self._code_stack[-1], CodeVars("")).variables)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Record attribute accesses on variables: x.foo, x.foo(), x.foo = val
        if isinstance(node.value, ast.Name):
            self._record_attribute_access(node.value.id, node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            dunder = {ast.Load: "__getitem__", ast.Store: "__setitem__", ast.Del: "__delitem__"}
            if (d := dunder.get(type(node.ctx))):
                self._record_attribute_access(node.value.id, d)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.left, ast.Name) and (d := _BINOP_TO_DUNDER.get(type(node.op))):
            self._record_attribute_access(node.left.id, d)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.operand, ast.Name) and (d := _UNARYOP_TO_DUNDER.get(type(node.op))):
            self._record_attribute_access(node.operand.id, d)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # x < y → __lt__ on x; item in x → __contains__ on x (right operand)
        prev = node.left
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if isinstance(comparator, ast.Name):
                    self._record_attribute_access(comparator.id, "__contains__")
            elif isinstance(prev, ast.Name) and (d := _CMPOP_TO_DUNDER.get(type(op))):
                self._record_attribute_access(prev.id, d)
            prev = comparator
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        if isinstance(node.value, ast.Name):
            self._record_attribute_access(node.value.id, "__await__")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Desugar builtin calls: len(x) → __len__ on x, etc.
        if (isinstance(node.func, ast.Name)
            and (dunder := _BUILTIN_TO_DUNDER.get(node.func.id))
            and self._is_builtin(node.func.id)
            and node.args
            and isinstance(node.args[0], ast.Name)):
            self._record_attribute_access(node.args[0].id, dunder)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        for gen in node.generators:
            if isinstance(gen.iter, ast.Name):
                self._record_attribute_access(gen.iter.id, "__iter__")
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        for gen in node.generators:
            if isinstance(gen.iter, ast.Name):
                self._record_attribute_access(gen.iter.id, "__iter__")
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for gen in node.generators:
            if isinstance(gen.iter, ast.Name):
                self._record_attribute_access(gen.iter.id, "__iter__")
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        for gen in node.generators:
            if isinstance(gen.iter, ast.Name):
                self._record_attribute_access(gen.iter.id, "__iter__")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # any NamedExpr within stay within the lambda's scope
        pass

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only pass value for simple single-target assignments like "x = None"
        # or "self.x = None". We skip multi-target (x = y = 1) and tuple
        # unpacking (a, b = 0, 1) to keep the implementation simple.
        value = node.value if (len(node.targets) == 1
                               and isinstance(node.targets[0], (ast.Name, ast.Attribute))) else None

        # Check if we're in a class body (not in a function)
        in_class_body = (
            self._class_stack
            and self._code_stack[-1] == self._class_stack[-1].name
        )

        for tgt in node.targets:
            for t in _iter_target_atoms(tgt):
                # Capture class-body assignments like `monitor = None`
                if in_class_body and isinstance(t, ast.Name):
                    class_info = self._class_stack[-1]
                    # Only record first assignment; capture initial constant type if applicable
                    if t.id not in class_info.class_attributes:
                        const_type = type(value.value) if (value is not None and isinstance(value, ast.Constant)) else None
                        class_info.class_attributes[t.id] = const_type
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
        if isinstance(node.target, ast.Name) and (d := _AUGASSIGN_TO_DUNDER.get(type(node.op))):
            self._record_attribute_access(node.target.id, d)
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if isinstance(node.iter, ast.Name):
            self._record_attribute_access(node.iter.id, "__iter__")
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        if isinstance(node.iter, ast.Name):
            self._record_attribute_access(node.iter.id, "__iter__")
        for t in _iter_target_atoms(node.target):
            self._record_target(t)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Name):
                self._record_attribute_access(item.context_expr.id, "__enter__")
                self._record_attribute_access(item.context_expr.id, "__exit__")
            if item.optional_vars is not None:
                for t in _iter_target_atoms(item.optional_vars):
                    self._record_target(t)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Name):
                self._record_attribute_access(item.context_expr.id, "__aenter__")
                self._record_attribute_access(item.context_expr.id, "__aexit__")
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

    result: dict[types.CodeType, CodeVars] = {}
    for co in qualname2code.values():
        accessed = f._accessed_attributes.get(co.co_qualname, {})
        if (codevars := f.code_vars.get(co.co_qualname)):
            if (scope_code := qualname2code.get(codevars.scope)):
                result[co] = replace(
                    codevars,
                    scope_code=scope_code,
                    class_key=qualname2code.get(codevars.class_name),
                    accessed_attributes=accessed,
                )
        elif accessed:
            # Function has attribute accesses but no local variables
            result[co] = CodeVars(
                scope=co.co_qualname,
                scope_code=co,
                accessed_attributes=accessed,
            )
    return result
