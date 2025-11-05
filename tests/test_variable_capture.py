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
            return set(codevars.variables.values()) | {v for v in codevars.attributes.values() if v is not None}
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
        """)
    m = map_variables(src)

    assert get(m, "C.__init__") == {"self.x"}
    assert get(m, "C.__init__.<locals>.f") == {"self.y"}
    assert get(m, "C.f") == {"me.x", "me.z"}
    assert get(m, "C.f2") == {"myself.x", "myself.y", "myself.z"}


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
