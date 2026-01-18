import textwrap
import pytest
import righttyper.variable_capture as variables
import ast
import types


def map_variables(source: str) -> dict[types.CodeType, variables.CodeVars]:
    tree = ast.parse(source)
    code = compile(tree, "<string>", "exec", optimize=0)
    return variables.map_variables(tree, code)


def get(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns a code->variables mapping by code name only."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            return set(codevars.variables.values()) | (
                {f"{codevars.self}.{attr}" for attr in codevars.attributes} if (codevars.attributes and codevars.self) else set()
            )
    return set()


def test_simple_assign_and_augassign():
    src = textwrap.dedent("""
        x = 1
        def f(y):
            a = 2
            a += 3
            y = 1
        """)

    m = map_variables(src)
    assert get(m, "<module>") == {"x"}
    assert get(m, "f") == {"a"}


def test_annassign_requires_value():
    src = textwrap.dedent("""
        x: int
        y: int = 3
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"y"}


def test_typealias():
    src = textwrap.dedent("""
        type x = int
        type y[T: int] = list[T]
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"x", "y"}


def test_unpacking_tuple_list_starred():
    src = textwrap.dedent("""
        (a, (b, *c)) = (1, (2, 3, 4))
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"a", "b", "c"}


def test_namedexpr_walrus_and_nested():
    src = textwrap.dedent("""
        def g():
            d = (e := 10)
            f = (h := (i := 1))
        """)
    m = map_variables(src)
    assert get(m, "g") == {"d", "e", "f", "h", "i"}


def test_attribute_chains_and_subscripts():
    src = textwrap.dedent("""
        x.y = 1
        z[2] = None
        """)
    m = map_variables(src)
    assert get(m, "<module>") == set()


def test_for_asyncfor_with_targets():
    src = textwrap.dedent("""
        async def a():
            async for (x, y) in agen():
                pass

        def b():
            for (u, v) in [(1,2)]:
                pass
        """)
    m = map_variables(src)
    assert {"x", "y"}.issubset(get(m, "a"))
    assert {"u", "v"}.issubset(get(m, "b"))


def test_with_and_asyncwith():
    src = textwrap.dedent("""
        from contextlib import asynccontextmanager, contextmanager

        @contextmanager
        def cm(): yield "ok"

        @asynccontextmanager
        async def acm(): yield "ok"

        def f():
            with cm() as w:
                pass

        async def g():
            async with acm() as aw:
                pass
        """)
    m = map_variables(src)
    assert "w" in get(m, "f")
    assert "aw" in get(m, "g")


def test_except_handler_name():
    src = textwrap.dedent("""
        def f():
            try:
                1/0
            except ZeroDivisionError as e:
                msg = str(e)
        """)
    m = map_variables(src)
    assert get(m, "f") == {"e", "msg"}


def test_import():
    src = textwrap.dedent("""
        import os as myos
        from math import sin as mysin, cos
        """)
    m = map_variables(src)
    assert get(m, "<module>") == set()


def test_class_and_method_bodies():
    src = textwrap.dedent("""
        class A:
            def __init__(myself):
                myself.a = None

        class C:
            y = 0
            def __init__(self):
                self.z = 3
                self.a = A()
                self.a.a = 1

            class D:
                z = "foo"

        c = C()
        c.z = 1
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"A", "C", "c"}

    assert get(m, "A") == set()
    assert get(m, "A.__init__") == {"myself.a"}

    assert get(m, "C") == {"C.y", "C.D"}
    assert get(m, "C.D") == {"C.D.z"}
    assert get(m, "C.__init__") == {"self.z", "self.a"}


def test_class_and_method_bodies_nested():
    src = textwrap.dedent("""
        def f():
            class C:
                y = 0
                def __init__(self):
                    self.z = 3

                class D:
                    z = "foo"
        """)
    m = map_variables(src)
    assert get(m, "<module>") == set()
    assert get(m, "f.<locals>.C") == {"C.y", "C.D"}
    assert get(m, "f.<locals>.C.D") == {"C.D.z"}
    assert get(m, "f.<locals>.C.__init__") == {"self.z"}


def test_self_not_in_method():
    src = textwrap.dedent("""
        def __init__(self):
            self.a = None

        class C:
            @staticmethod
            def foo(self):  # not the self you want
                self.x = 0

            @classmethod
            def bar(self):  # class, not self
                self.y = 0
        """)
    m = map_variables(src)

    assert get(m, "__init__") == set()
    assert get(m, "C.foo") == set()
    assert get(m, "C.bar") == set()


def test_self_nested_in_method():
    src = textwrap.dedent("""
        class C:
            def f1(self):
                def f():
                    def g():
                        self.x = 0

            def f2(self):
                def f():
                    self = None
                    def g():
                        self.y = 0

            def f3(self):
                def f(self):
                    def g():
                        self.z = 0
        """)
    m = map_variables(src)

    assert get(m, "C.f1.<locals>.f.<locals>.g") == {"self.x"}
    assert get(m, "C.f2.<locals>.f.<locals>.g") == set()
    assert get(m, "C.f3.<locals>.f.<locals>.g") == set()


def test_self_attribute_in_various_methods():
    src = textwrap.dedent("""
        class C:
            def __init__(self):
                self.x = 1
                def f():
                    self.y = 2

            def f(me):
                me.x = 1
                me.z = 3
                self.y = None   # an error

            def f2(myself):
                myself.x = 1
                myself.y = 2
                myself.z = 3

            def f3(self):
                pass
        """)
    m = map_variables(src)

    assert get(m, "C.__init__") == {"self.x", "self.y", "self.z"}
    assert get(m, "C.__init__.<locals>.f") == {"self.x", "self.y", "self.z"}
    assert get(m, "C.f") == {"me.x", "me.y", "me.z"}
    assert get(m, "C.f2") == {"myself.x", "myself.y", "myself.z"}
    assert get(m, "C.f3") == {"self.x", "self.y", "self.z"}


def test_lambdas_and_comprehensions_have_their_own_code_objects():
    src = textwrap.dedent("""
        L = [(a:=w) for w in range(2)]
        S = {(b:=w) for w in range(2)}
        D = {k: (c:=v) for k, v in [(1,2)]}
        G = ((d:=x) for x in range(3)); next(G)
        h = lambda q: (e:=q+1); h(0)
        """)
    mapping = map_variables(src)
    assert get(mapping, "<module>") == {"L", "S", "D", "G", "h",
                                         "a", "b", "c", "d"}
    assert get(mapping, "<genexpr>") == set()
    assert get(mapping, "<lambda>") == set()


def test_match_tuple_pattern():
    src = textwrap.dedent("""
        def f(data):
            match data:
                case (x, y):
                    z = x + y
        """)
    m = map_variables(src)
    assert get(m, "f") == {"x", "y", "z"}


def test_match_class_pattern_with_fields():
    src = textwrap.dedent("""
        def f(obj):
            match obj:
                case Point(x=a, y=b):
                    total = a + b
        """)
    m = map_variables(src)
    assert get(m, "f") == {"a", "b", "total"}


def test_match_nested_mapping_and_sequence():
    src = textwrap.dedent("""
        def f(record):
            match record:
                case ("user", {"id": uid, "name": name}):
                    seen = {uid, name}
        """)
    m = map_variables(src)
    assert get(m, "f") == {"uid", "name", "seen"}


def test_match_sequence_starred_and_count():
    src = textwrap.dedent("""
        def f(seq):
            match seq:
                case [first, *rest]:
                    count = len(rest)
        """)
    m = map_variables(src)
    assert get(m, "f") == {"first", "rest", "count"}


def test_match_or_requires_same_bindings():
    src = textwrap.dedent("""
        def f(value):
            match value:
                case ("ok", n) | ("warn", n):
                    result = n * 2
        """)
    m = map_variables(src)
    assert get(m, "f") == {"n", "result"}


def test_match_mapping_with_rest():
    src = textwrap.dedent("""
        def f(d):
            match d:
                case {"x": x, **rest}:
                    pass
        """)
    m = map_variables(src)
    assert get(m, "f") == {"x", "rest"}


def test_match_class_pattern_with_as_binding():
    src = textwrap.dedent("""
        def f(node):
            match node:
                case Tree(left=l, right=r) as tree:
                    depth = max(l, r)
        """)
    m = map_variables(src)
    assert get(m, "f") == {"l", "r", "tree", "depth"}


def test_match_as_simple_name_only():
    src = textwrap.dedent("""
        def f(x):
            match x:
                case y:
                    z = y
        """)
    m = map_variables(src)
    assert get(m, "f") == {"y", "z"}


def get_initial_constants(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns initial_constants mapping by code name only."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            return codevars.initial_constants
    return {}


def test_initial_constant_none_captured():
    """Test that x = None captures NoneType as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = None
            x = 5
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == type(None)


def test_initial_constant_int_captured():
    """Test that x = 0 captures int as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = 0
            x = "string"
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == int


def test_initial_constant_str_captured():
    """Test that x = "" captures str as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = "hello"
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == str


def test_non_constant_not_captured():
    """Test that x = [] does NOT capture (not ast.Constant)."""
    src = textwrap.dedent("""
        def foo():
            x = []
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert "x" not in consts


def test_function_call_not_captured():
    """Test that x = foo() does NOT capture (not ast.Constant)."""
    src = textwrap.dedent("""
        def foo():
            x = some_func()
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert "x" not in consts


def test_reassignment_preserves_initial():
    """Test that reassignment doesn't overwrite initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = None
            x = 5
            x = "hello"
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    # Should be NoneType (first assignment), not int or str
    assert consts.get("x") == type(None)


def test_initial_constant_bool_captured():
    """Test that x = True captures bool as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = True
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == bool


def test_initial_constant_float_captured():
    """Test that x = 3.14 captures float as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = 3.14
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == float


def test_module_level_initial_constant():
    """Test initial constants at module level."""
    src = textwrap.dedent("""
        x = None
        y = 42
        z = []
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "<module>")
    assert consts.get("x") == type(None)
    assert consts.get("y") == int
    assert "z" not in consts


def test_annotated_assignment_initial_constant():
    """Test that x: int = None captures NoneType."""
    src = textwrap.dedent("""
        def foo():
            x: int = None
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == type(None)


def test_declaration_without_assignment_not_captured():
    """Test that x: str (no value) does NOT capture anything from declaration."""
    src = textwrap.dedent("""
        def foo():
            x: str
            x = 5
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    # The declaration has no value, so no initial constant from it
    # The subsequent assignment x = 5 should capture int (not str from declaration)
    assert consts.get("x") == int


def test_tuple_unpacking_not_captured():
    """Test that a, b = 0, 1 does NOT capture (can't match values to targets)."""
    src = textwrap.dedent("""
        def foo():
            a, b = 0, 1
            return a + b
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    # Tuple unpacking is too complex to match values to targets
    assert "a" not in consts
    assert "b" not in consts


def get_attribute_initial_constants(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns attribute initial_constants for a method by name."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            return codevars.attribute_initial_constants if codevars.attribute_initial_constants else {}
    return {}


def test_attribute_initial_constant_none_captured():
    """Test that self.x = None captures NoneType for attribute."""
    src = textwrap.dedent("""
        class C:
            def __init__(self):
                self.x = None
                self.x = some_value
        """)
    m = map_variables(src)
    # Attribute constants are on the method's CodeVars (shared from ClassInfo)
    consts = get_attribute_initial_constants(m, "C.__init__")
    assert consts.get("x") == type(None)


def test_attribute_initial_constant_int_captured():
    """Test that self.x = 0 captures int for attribute."""
    src = textwrap.dedent("""
        class C:
            def __init__(self):
                self.x = 0
        """)
    m = map_variables(src)
    consts = get_attribute_initial_constants(m, "C.__init__")
    assert consts.get("x") == int


def test_attribute_non_constant_not_captured():
    """Test that self.x = foo() does NOT capture."""
    src = textwrap.dedent("""
        class C:
            def __init__(self):
                self.x = foo()
        """)
    m = map_variables(src)
    consts = get_attribute_initial_constants(m, "C.__init__")
    assert "x" not in consts
