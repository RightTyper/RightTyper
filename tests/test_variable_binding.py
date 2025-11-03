import textwrap
import pytest
import libcst as cst
from libcst.metadata import MetadataWrapper, QualifiedNameProvider, PositionProvider
from libcst.helpers import get_full_name_for_node
from righttyper.variable_binding import VariableBindingProvider


def get_first_defs(source: str) -> dict[str, str]:
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    mapping = wrapper.resolve(VariableBindingProvider)

    class QNames(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (QualifiedNameProvider, PositionProvider)

        def __init__(self) -> None:
            self.node_name: dict[cst.CSTNode, str] = {}

        def on_visit(self, node: cst.CSTNode) -> bool:
            try:
                qnames = self.get_metadata(QualifiedNameProvider, node)
                self.node_name[node] = next(qn.name for qn in qnames)
            except:
                pass
            return True

    qnames = QNames()
    wrapper.visit(qnames)

    def class_name(qualname):
        qualname = list(qualname)
        while qualname[-1] == '<locals>':
            qualname.pop()
            qualname.pop()
        return qualname

    return {
        '.'.join((*result.qualname, name)): cst.Module([]).code_for_node(node).strip()
        for node, result in mapping.items()
        for name in result.defines_vars
    } | {
        '.'.join((*class_name(result.qualname), result.self_name, name)): cst.Module([]).code_for_node(node).strip()
        for node, result in mapping.items()
        for name in result.defines_attrs
    }


def test_assign():
    src = textwrap.dedent("""
        a = 1
        a += 1
        a, *b = 2, 3, 4
        d: int
        b = c = d = 5
        """)

    m = get_first_defs(src)
    assert m['a'] == 'a = 1'
    assert m['b'] == 'a, *b = 2, 3, 4'
    assert m['c'] == 'b = c = d = 5'
    assert m['d'] == 'd: int'


def test_namedexpr():
    src = textwrap.dedent("""
        a = (b := 1)
        a = None
        b = None
        """)

    m = get_first_defs(src)
    assert m['a'] == 'a = (b := 1)'
    assert m['b'] == '(b := 1)'


def test_function():
    src = textwrap.dedent("""
        a = 42
        def f():
            a = 1
            a += 1
            a, *b = 2, 3, 4
            d: int
            b = c = d = 5
        """)

    m = get_first_defs(src)
    assert m['a'] == 'a = 42'
    assert m['f.<locals>.a'] == 'a = 1'
    assert m['f.<locals>.b'] == 'a, *b = 2, 3, 4'
    assert m['f.<locals>.c'] == 'b = c = d = 5'
    assert m['f.<locals>.d'] == 'd: int'


def test_function_arg():
    src = textwrap.dedent("""
        def f(x: int):
            x = 42
        """)

    m = get_first_defs(src)
    assert m['f.<locals>.x'] == 'x: int'


def test_attribute_chains_and_subscripts():
    src = textwrap.dedent("""
        x.y = 1
        z[2] = None
        """)
    m = get_first_defs(src)
    assert 'x.y' not in m
    assert 'z' not in m


def test_for_asyncfor_with_targets():
    src = textwrap.dedent("""
        async def a():
            async for (x, y) in agen():
                x = y = 0
                pass

        def b():
            for (u, v) in [(1,2)]:
                u = v = 0
                pass
        """)
    m = get_first_defs(src)
    assert m['a.<locals>.x'].startswith('async for')
    assert m['a.<locals>.y'].startswith('async for')
    assert m['b.<locals>.u'].startswith('for')
    assert m['b.<locals>.v'].startswith('for')


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

    m = get_first_defs(src)
    assert m['f.<locals>.w'].startswith('with cm() as w:')
    assert m['g.<locals>.aw'].startswith('async with acm() as aw:')


def test_except_handler_name():
    src = textwrap.dedent("""
        def f():
            try:
                1/0
            except ZeroDivisionError as e:
                pass
            e = None
        """)
    m = get_first_defs(src)
    assert m['f.<locals>.e'].startswith('except ZeroDivisionError as e:')


@pytest.mark.parametrize("from_", ["", "from zoo "])
def test_import(from_):
    src = textwrap.dedent(f"""\
        import rhino.cerous
        {from_}import elefant as efelant, cow as cat

        rhino = 1
        efelant = 2
        """)

    m = get_first_defs(src)
    assert 'import rhino' in m['rhino']
    assert 'rhino.cerous' not in m
    assert 'as efelant' in m['efelant']
    assert 'as cat' in m['cat']
    assert 'cow' not in m


def test_import_star():
    src = textwrap.dedent("""\
        from zoo import *
        """)

    m = get_first_defs(src)
    assert not m


def test_class_and_method_bodies():
    src = textwrap.dedent("""
        class A:
            def __init__(myself):
                self.a = "wrong"
                myself.a = None

        class C:
            y = 0
            def __init__(me):
                me.z = 3
                me.a = A()
                me.a.a = 1

            def f(myself):
                myself.a = None

            class D:
                z = "foo"

        c = C()
        c.z = 1
        """)

    m = get_first_defs(src)
    assert m['A'].startswith('class A')
    assert m['C'].startswith('class C')
    assert m['c'] == "c = C()"

    assert m['A.myself.a'] == 'myself.a = None'

    assert m['C.y'] == 'y = 0'

    assert m['C.me.z'] == 'me.z = 3'
    assert m['C.me.a'] == 'me.a = A()'

    assert m['C.D.z'] == 'z = "foo"'


def test_self_not_in_method():
    src = textwrap.dedent("""
        def __init__(self):
            self.a = None

        class C:
            @staticmethod
            def foo(self):  # not the self you are looking for
                self.x = 0

            @classmethod
            def bar(self):  # class, not self
                self.y = 0
        """)
    m = get_first_defs(src)

    assert not any("self.a" in k for k in m)
    assert "C.self.x" not in m
    assert "C.self.y" not in m


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
    m = get_first_defs(src)
    assert m['f.<locals>.C'].startswith("class C")
    assert m['f.<locals>.C.y'] == 'y = 0'
    assert m['f.<locals>.C.self.z'] == 'self.z = 3'
    assert m['f.<locals>.C.D.z'] == 'z = "foo"'


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
    m = get_first_defs(src)
    assert m['C.self.x'] == 'self.x = 0'
    assert 'C.self.y' not in m
    assert 'C.self.z' not in m


def test_lambdas_and_comprehensions_definitions():
    src = textwrap.dedent("""
        L = [(a:=w) for w in range(2)]
        S = {(b:=w) for w in range(2)}
        D = {k: (c:=v) for k, v in [(1,2)]}
        G = ((d:=x) for x in range(3)); next(G)
        h = lambda q: (e:=q+1); h(0)
        """)
    m = get_first_defs(src)
    assert m['a'] == '(a:=w)'
    assert m['b'] == '(b:=w)'
    assert m['c'] == '(c:=v)'
    assert m['d'] == '(d:=x)'
    assert 'e' not in m


def test_match_tuple_pattern():
    src = textwrap.dedent("""
        def f(data):
            match data:
                case (x, y):
                    z = x + y
        """)
    m = get_first_defs(src)
    assert m['f.<locals>.x'].startswith('case (x, y)')
    assert m['f.<locals>.y'].startswith('case (x, y)')
    assert 'f.<locals>.z' in m


def test_match_class_pattern_with_fields():
    src = textwrap.dedent("""
        def f(obj):
            match obj:
                case Point(x=a, y=b):
                    total = a + b
        """)
    m = get_first_defs(src)
    assert m['f.<locals>.a'].startswith('case Point')
    assert m['f.<locals>.b'].startswith('case Point')
    assert 'f.<locals>.total' in m


def test_match_nested_mapping_and_sequence():
    src = textwrap.dedent("""
        def f(record):
            match record:
                case ("user", {"id": uid, "name": name}):
                    seen = {uid, name}
        """)
    m = get_first_defs(src)
    assert m['f.<locals>.uid'].startswith('case ("user')
    assert m['f.<locals>.name'].startswith('case ("user')
    assert 'f.<locals>.seen' in m


def test_match_sequence_starred_and_count():
    src = textwrap.dedent("""
        def f(seq):
            match seq:
                case [first, *rest]:
                    count = len(rest)
        """)
    m = get_first_defs(src)
    assert m['f.<locals>.first'].startswith('case [first')
    assert m['f.<locals>.rest'].startswith('case [first')
    assert 'f.<locals>.count' in m


def test_match_or_requires_same_bindings():
    src = textwrap.dedent("""
        def f(value):
            match value:
                case ("ok", n) | ("warn", n):
                    result = n * 2
        """)
    m = get_first_defs(src)
    assert 'f.<locals>.n' in m
    assert 'f.<locals>.result' in m


def test_match_mapping_with_rest():
    src = textwrap.dedent("""
        def f(d):
            match d:
                case {"x": x, **rest}:
                    pass
        """)
    m = get_first_defs(src)
    assert 'f.<locals>.x' in m
    assert 'f.<locals>.rest' in m


def test_match_class_pattern_with_as_binding():
    src = textwrap.dedent("""
        def f(node):
            match node:
                case Tree(left=l, right=r) as tree:
                    depth = max(l, r)
        """)
    m = get_first_defs(src)
    assert 'f.<locals>.l' in m
    assert 'f.<locals>.r' in m
    assert 'f.<locals>.depth' in m


def test_match_as_simple_name_only():
    src = textwrap.dedent("""
        def f(x):
            match x:
                case y:
                    z = y
        """)
    m = get_first_defs(src)
    assert 'f.<locals>.y' in m
    assert 'f.<locals>.z' in m


def test_nonlocal_and_global():
    src = textwrap.dedent("""
        global a

        def f(x):
            b = 1

            def g(y):
                global a
                nonlocal b
                a = 2
                b = 3

            g(0)

        a = 4
        """)

    m = get_first_defs(src)
    assert m['a'] == 'a = 4'
    assert m['f.<locals>.b'] == 'b = 1'
    assert 'f.<locals>.g.<locals>.a' not in m
    assert 'f.<locals>.g.<locals>.b' not in m
