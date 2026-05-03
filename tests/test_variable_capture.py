import textwrap
import pytest
import righttyper.variable_capture as variables
import ast
import types


@pytest.fixture(params=[True, False], ids=['track_attrs', 'no_track_attrs'])
def track_attributes(request):
    return request.param


@pytest.fixture
def map_variables(track_attributes):
    def _impl(source: str) -> dict[types.CodeType, variables.CodeVars]:
        tree = ast.parse(source)
        code = compile(tree, "<string>", "exec", optimize=0)
        return variables.map_variables(tree, code, track_attributes=track_attributes)
    return _impl


def get(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns qualified variable names by code name."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            prefix = codevars.var_prefix
            return {f"{prefix}{var}" for var in codevars.variables.keys()} | (
                {f"{codevars.self}.{attr}" for attr in codevars.attributes} if (codevars.attributes and codevars.self) else set()
            )
    return set()


def test_simple_assign_and_augassign(map_variables):
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


def test_annassign_requires_value(map_variables):
    src = textwrap.dedent("""
        x: int
        y: int = 3
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"y"}


def test_typealias(map_variables):
    src = textwrap.dedent("""
        type x = int
        type y[T: int] = list[T]
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"x", "y"}


def test_unpacking_tuple_list_starred(map_variables):
    src = textwrap.dedent("""
        (a, (b, *c)) = (1, (2, 3, 4))
        """)
    m = map_variables(src)
    assert get(m, "<module>") == {"a", "b", "c"}


def test_namedexpr_walrus_and_nested(map_variables):
    src = textwrap.dedent("""
        def g():
            d = (e := 10)
            f = (h := (i := 1))
        """)
    m = map_variables(src)
    assert get(m, "g") == {"d", "e", "f", "h", "i"}


def test_attribute_chains_and_subscripts(map_variables):
    src = textwrap.dedent("""
        x.y = 1
        z[2] = None
        """)
    m = map_variables(src)
    assert get(m, "<module>") == set()


def test_for_asyncfor_with_targets(map_variables):
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


def test_with_and_asyncwith(map_variables):
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


def test_except_handler_name(map_variables):
    src = textwrap.dedent("""
        def f():
            try:
                1/0
            except ZeroDivisionError as e:
                msg = str(e)
        """)
    m = map_variables(src)
    assert get(m, "f") == {"e", "msg"}


def test_import(map_variables):
    src = textwrap.dedent("""
        import os as myos
        from math import sin as mysin, cos
        """)
    m = map_variables(src)
    assert get(m, "<module>") == set()


def test_class_and_method_bodies(map_variables):
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


def test_class_and_method_bodies_nested(map_variables):
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


def test_self_not_in_method(map_variables):
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


def test_self_nested_in_method(map_variables):
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


def test_self_attribute_in_various_methods(map_variables):
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


def test_lambdas_and_comprehensions_have_their_own_code_objects(map_variables):
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


def test_match_tuple_pattern(map_variables):
    src = textwrap.dedent("""
        def f(data):
            match data:
                case (x, y):
                    z = x + y
        """)
    m = map_variables(src)
    assert get(m, "f") == {"x", "y", "z"}


def test_match_class_pattern_with_fields(map_variables):
    src = textwrap.dedent("""
        def f(obj):
            match obj:
                case Point(x=a, y=b):
                    total = a + b
        """)
    m = map_variables(src)
    assert get(m, "f") == {"a", "b", "total"}


def test_match_nested_mapping_and_sequence(map_variables):
    src = textwrap.dedent("""
        def f(record):
            match record:
                case ("user", {"id": uid, "name": name}):
                    seen = {uid, name}
        """)
    m = map_variables(src)
    assert get(m, "f") == {"uid", "name", "seen"}


def test_match_sequence_starred_and_count(map_variables):
    src = textwrap.dedent("""
        def f(seq):
            match seq:
                case [first, *rest]:
                    count = len(rest)
        """)
    m = map_variables(src)
    assert get(m, "f") == {"first", "rest", "count"}


def test_match_or_requires_same_bindings(map_variables):
    src = textwrap.dedent("""
        def f(value):
            match value:
                case ("ok", n) | ("warn", n):
                    result = n * 2
        """)
    m = map_variables(src)
    assert get(m, "f") == {"n", "result"}


def test_match_mapping_with_rest(map_variables):
    src = textwrap.dedent("""
        def f(d):
            match d:
                case {"x": x, **rest}:
                    pass
        """)
    m = map_variables(src)
    assert get(m, "f") == {"x", "rest"}


def test_match_class_pattern_with_as_binding(map_variables):
    src = textwrap.dedent("""
        def f(node):
            match node:
                case Tree(left=l, right=r) as tree:
                    depth = max(l, r)
        """)
    m = map_variables(src)
    assert get(m, "f") == {"l", "r", "tree", "depth"}


def test_match_as_simple_name_only(map_variables):
    src = textwrap.dedent("""
        def f(x):
            match x:
                case y:
                    z = y
        """)
    m = map_variables(src)
    assert get(m, "f") == {"y", "z"}


def get_initial_constants(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns initial constants mapping by code name (variables where value is not None)."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            return {k: v for k, v in codevars.variables.items() if v is not None}
    return {}


def test_initial_constant_none_captured(map_variables):
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


def test_initial_constant_int_captured(map_variables):
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


def test_initial_constant_str_captured(map_variables):
    """Test that x = "" captures str as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = "hello"
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == str


def test_non_constant_not_captured(map_variables):
    """Test that x = [] does NOT capture (not ast.Constant)."""
    src = textwrap.dedent("""
        def foo():
            x = []
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert "x" not in consts


def test_function_call_not_captured(map_variables):
    """Test that x = foo() does NOT capture (not ast.Constant)."""
    src = textwrap.dedent("""
        def foo():
            x = some_func()
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert "x" not in consts


def test_reassignment_preserves_initial(map_variables):
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


def test_initial_constant_bool_captured(map_variables):
    """Test that x = True captures bool as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = True
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == bool


def test_initial_constant_float_captured(map_variables):
    """Test that x = 3.14 captures float as initial constant."""
    src = textwrap.dedent("""
        def foo():
            x = 3.14
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == float


def test_module_level_initial_constant(map_variables):
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


def test_annotated_assignment_initial_constant(map_variables):
    """Test that x: int = None captures NoneType."""
    src = textwrap.dedent("""
        def foo():
            x: int = None
            return x
        """)
    m = map_variables(src)
    consts = get_initial_constants(m, "foo")
    assert consts.get("x") == type(None)


def test_declaration_without_assignment_not_captured(map_variables):
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


def test_tuple_unpacking_not_captured(map_variables):
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
            if codevars.attributes:
                return {k: v for k, v in codevars.attributes.items() if v is not None}
            return {}
    return {}


def test_attribute_initial_constant_none_captured(map_variables):
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


def test_attribute_initial_constant_int_captured(map_variables):
    """Test that self.x = 0 captures int for attribute."""
    src = textwrap.dedent("""
        class C:
            def __init__(self):
                self.x = 0
        """)
    m = map_variables(src)
    consts = get_attribute_initial_constants(m, "C.__init__")
    assert consts.get("x") == int


def test_attribute_non_constant_not_captured(map_variables):
    """Test that self.x = foo() does NOT capture."""
    src = textwrap.dedent("""
        class C:
            def __init__(self):
                self.x = foo()
        """)
    m = map_variables(src)
    consts = get_attribute_initial_constants(m, "C.__init__")
    assert "x" not in consts


# =============================================================================
# Tests for class attribute capture (cls.x and class body assignments)
# =============================================================================

def get_class_attributes(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns class_attributes as a set of attribute names by code name."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            return set(codevars.class_attributes.keys()) if codevars.class_attributes else set()
    return set()


def get_class_attribute_initial_constants(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns class_attribute initial constants mapping by code name."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            if codevars.class_attributes:
                return {k: v for k, v in codevars.class_attributes.items() if v is not None}
            return {}
    return {}


def test_class_body_attribute_captured(map_variables):
    """Class body assignment like `monitor = None` captures class attribute."""
    src = textwrap.dedent("""
        class C:
            monitor = None
        """)
    m = map_variables(src)
    # Class attributes should be visible from methods in the class
    # For now, check the class code object itself
    attrs = get_class_attributes(m, "C")
    assert "monitor" in attrs


def test_class_body_attribute_initial_constant(map_variables):
    """Class body assignment `x = None` captures NoneType as initial constant."""
    src = textwrap.dedent("""
        class C:
            x = None
        """)
    m = map_variables(src)
    consts = get_class_attribute_initial_constants(m, "C")
    assert consts.get("x") == type(None)


def test_class_body_non_constant_not_captured(map_variables):
    """Class body assignment `x = []` does NOT capture initial constant."""
    src = textwrap.dedent("""
        class C:
            x = []
        """)
    m = map_variables(src)
    consts = get_class_attribute_initial_constants(m, "C")
    assert "x" not in consts


def test_cls_assignment_captured(map_variables):
    """cls.x = value in classmethod captures class attribute."""
    src = textwrap.dedent("""
        class C:
            @classmethod
            def setup(cls):
                cls.monitor = None
        """)
    m = map_variables(src)
    # The classmethod should have access to class_attributes
    attrs = get_class_attributes(m, "C.setup")
    assert "monitor" in attrs


def test_cls_assignment_initial_constant(map_variables):
    """cls.x = None in classmethod captures NoneType as initial constant."""
    src = textwrap.dedent("""
        class C:
            @classmethod
            def setup(cls):
                cls.x = None
        """)
    m = map_variables(src)
    consts = get_class_attribute_initial_constants(m, "C.setup")
    assert consts.get("x") == type(None)


def test_cls_assignment_non_constant(map_variables):
    """cls.x = foo() in classmethod does NOT capture initial constant."""
    src = textwrap.dedent("""
        class C:
            @classmethod
            def setup(cls):
                cls.x = foo()
        """)
    m = map_variables(src)
    consts = get_class_attribute_initial_constants(m, "C.setup")
    assert "x" not in consts


def test_class_body_and_cls_combined(map_variables):
    """Class body + classmethod assignments both contribute to class_attributes."""
    src = textwrap.dedent("""
        class C:
            monitor = None

            @classmethod
            def setup(cls):
                cls.other = 42
        """)
    m = map_variables(src)
    # Class body attribute
    class_attrs = get_class_attributes(m, "C")
    assert "monitor" in class_attrs
    # Classmethod attribute (shared from ClassInfo)
    method_attrs = get_class_attributes(m, "C.setup")
    assert "other" in method_attrs
    # Both should be visible since they're on the same ClassInfo
    assert "monitor" in method_attrs


def test_cls_parameter_tracked(map_variables):
    """The cls parameter is tracked for classmethods."""
    src = textwrap.dedent("""
        class C:
            @classmethod
            def setup(cls):
                cls.x = 1
        """)
    m = map_variables(src)
    # Check that CodeVars has cls field set
    for co, codevars in m.items():
        if co.co_qualname == "C.setup":
            assert codevars.cls == "cls"
            break
    else:
        assert False, "C.setup not found"


def test_staticmethod_no_cls(map_variables):
    """Static methods should not have cls tracked."""
    src = textwrap.dedent("""
        class C:
            @staticmethod
            def helper():
                pass
        """)
    m = map_variables(src)
    for co, codevars in m.items():
        if co.co_qualname == "C.helper":
            assert codevars.cls is None
            assert codevars.self is None
            break
    else:
        assert False, "C.helper not found"


def test_regular_method_no_cls(map_variables):
    """Regular methods should have self but not cls."""
    src = textwrap.dedent("""
        class C:
            def method(self):
                pass
        """)
    m = map_variables(src)
    for co, codevars in m.items():
        if co.co_qualname == "C.method":
            assert codevars.self == "self"
            assert codevars.cls is None
            break
    else:
        assert False, "C.method not found"


# =============================================================================
# Tests for accessed attribute collection
# =============================================================================

def get_accessed_attributes(mapping: dict[types.CodeType, variables.CodeVars], name: str):
    """Returns accessed_attributes by code name."""
    for co, codevars in mapping.items():
        if co.co_qualname == name:
            return codevars.accessed_attributes
    return {}


def test_accessed_attributes_simple(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x.foo collects 'foo' on parameter x."""
    src = textwrap.dedent("""
        def f(x):
            return x.name
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert attrs.get("x") == {"name"}


def test_accessed_attributes_method_call(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x.foo() still collects 'foo' — it's an attribute access followed by a call."""
    src = textwrap.dedent("""
        def f(x):
            return x.process()
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert attrs.get("x") == {"process"}


def test_accessed_attributes_multiple_vars(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """Different attributes on different variables are tracked separately."""
    src = textwrap.dedent("""
        def f(x, y):
            return x.name + y.value
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert attrs.get("x") == {"name"}
    assert attrs.get("y") == {"value"}


def test_accessed_attributes_multiple_on_same_var(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """Multiple attributes on the same variable are all collected."""
    src = textwrap.dedent("""
        def f(x):
            return x.name + x.value
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert attrs.get("x") == {"name", "value"}


def test_accessed_attributes_chained(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x.foo.bar collects 'foo' on x (bar is on foo's result, not x)."""
    src = textwrap.dedent("""
        def f(x):
            return x.parent.name
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert attrs.get("x") == {"parent"}
    assert "name" not in attrs.get("x", set())


def test_accessed_attributes_includes_writes(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x.foo = val is still an attribute access — the type needs to support it."""
    src = textwrap.dedent("""
        def f(x):
            x.name = "hello"
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "name" in attrs.get("x", set())


def test_accessed_attributes_locals(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """Attribute access on local variables, not just parameters."""
    src = textwrap.dedent("""
        def f():
            result = get_result()
            return result.count
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert attrs.get("result") == {"count"}


def test_accessed_attributes_alias(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """y = x; y.name records 'name' on x."""
    src = textwrap.dedent("""
        def f(x):
            y = x
            return y.name
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "name" in attrs.get("x", set())


def test_accessed_attributes_transitive_alias(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """z = y; y = x; z still aliases through to x."""
    src = textwrap.dedent("""
        def f(x):
            y = x
            z = y
            return z.name
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "name" in attrs.get("x", set())


def test_accessed_attributes_alias_reassigned(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """Reassignment after alias: conservatively keeps the alias (could be a branch)."""
    src = textwrap.dedent("""
        def f(x):
            y = x
            y = something_else()
            return y.name
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    # Without flow analysis, we can't distinguish sequential reassignment from
    # branches, so we conservatively keep the alias. This over-approximates
    # (records 'name' on x even though y was reassigned), which is safe —
    # it just makes simplification slightly more conservative.
    assert "name" in attrs.get("x", set())


def test_accessed_attributes_alias_branch_conservative(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """Aliases from different branches are unioned (conservative)."""
    src = textwrap.dedent("""
        def f(x, y):
            if True:
                z = x
            else:
                z = y
            return z.value
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    # z could be x or y, so 'value' should be recorded on both
    assert "value" in attrs.get("x", set())
    assert "value" in attrs.get("y", set())


def test_aliases_do_not_leak_across_sibling_functions(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """An alias recorded in one function (y = x) must not be visible to a
    sibling function's same-named variable. With shared alias state, h's
    y.foo would erroneously be recorded as foo on x — even though h's y is
    its own parameter, unrelated to f's y."""
    src = textwrap.dedent("""
        def f(x):
            y = x
            return y.shared

        def h(y):
            return y.h_only
        """)
    m = map_variables(src)
    f_attrs = get_accessed_attributes(m, "f")
    h_attrs = get_accessed_attributes(m, "h")

    # f sees 'shared' on x via the alias y = x.
    assert "shared" in f_attrs.get("x", set())
    # h's y.h_only must not leak to f's x.
    assert "h_only" not in f_attrs.get("x", set()), (
        f"h's y.h_only leaked to f's x via shared alias state: {f_attrs}"
    )
    # And h itself records h_only on its own y.
    assert "h_only" in h_attrs.get("y", set())


def test_aliases_propagate_into_nested_functions(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """Closures should still see their enclosing scope's aliases."""
    src = textwrap.dedent("""
        def f(x):
            y = x
            def inner():
                return y.name
            return inner
        """)
    m = map_variables(src)
    f_attrs = get_accessed_attributes(m, "f")
    inner_attrs = get_accessed_attributes(m, "f.<locals>.inner")
    # inner's access on y should propagate to x via f's alias.
    assert "name" in f_attrs.get("x", set()) or "name" in inner_attrs.get("x", set())


def test_accessed_attributes_non_name_assignment_no_alias(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """y = x.child is not an alias — attribute access on y stays on y."""
    src = textwrap.dedent("""
        def f(x):
            y = x.child
            return y.name
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    # y is not an alias for x — y.name is on y, and x.child is on x
    assert attrs.get("y") == {"name"}
    assert attrs.get("x") == {"child"}


# =============================================================================
# Tests for operator/builtin desugaring to dunder attributes
# =============================================================================

def test_desugar_subscript_getitem(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x[i] → __getitem__ on x."""
    src = textwrap.dedent("""
        def f(x, i):
            return x[i]
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__getitem__" in attrs.get("x", set())


def test_desugar_subscript_setitem(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x[i] = v → __setitem__ on x."""
    src = textwrap.dedent("""
        def f(x, i, v):
            x[i] = v
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__setitem__" in attrs.get("x", set())


def test_desugar_subscript_delitem(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """del x[i] → __delitem__ on x."""
    src = textwrap.dedent("""
        def f(x, i):
            del x[i]
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__delitem__" in attrs.get("x", set())


def test_desugar_for_iter(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """for item in x → __iter__ on x."""
    src = textwrap.dedent("""
        def f(x):
            for item in x:
                pass
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__iter__" in attrs.get("x", set())


def test_desugar_comprehension_iter(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """[... for item in x] → __iter__ on x."""
    src = textwrap.dedent("""
        def f(x):
            return [item for item in x]
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__iter__" in attrs.get("x", set())


def test_desugar_binop(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x + y → __add__ on x."""
    src = textwrap.dedent("""
        def f(x, y):
            return x + y
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__add__" in attrs.get("x", set())


def test_desugar_augassign(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x += y → __iadd__ on x."""
    src = textwrap.dedent("""
        def f(x, y):
            x += y
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__iadd__" in attrs.get("x", set())


def test_desugar_unaryop(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """-x → __neg__ on x."""
    src = textwrap.dedent("""
        def f(x):
            return -x
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__neg__" in attrs.get("x", set())


def test_desugar_compare(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """x < y → __lt__ on x."""
    src = textwrap.dedent("""
        def f(x, y):
            return x < y
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__lt__" in attrs.get("x", set())


def test_desugar_contains(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """item in x → __contains__ on x (right operand)."""
    src = textwrap.dedent("""
        def f(item, x):
            return item in x
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__contains__" in attrs.get("x", set())


def test_desugar_with(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """with x: → __enter__ and __exit__ on x."""
    src = textwrap.dedent("""
        def f(x):
            with x:
                pass
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__enter__" in attrs.get("x", set())
    assert "__exit__" in attrs.get("x", set())


def test_desugar_builtin_len(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """len(x) → __len__ on x."""
    src = textwrap.dedent("""
        def f(x):
            return len(x)
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__len__" in attrs.get("x", set())


def test_desugar_builtin_iter(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """iter(x) → __iter__ on x."""
    src = textwrap.dedent("""
        def f(x):
            return iter(x)
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__iter__" in attrs.get("x", set())


def test_desugar_builtin_reversed(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """reversed(x) → __reversed__ on x."""
    src = textwrap.dedent("""
        def f(x):
            return reversed(x)
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__reversed__" in attrs.get("x", set())


def test_desugar_builtin_shadowed(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """If len is shadowed by a local, don't desugar."""
    src = textwrap.dedent("""
        def f(x):
            len = lambda y: 42
            return len(x)
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__len__" not in attrs.get("x", set())


def test_desugar_builtin_shadowed_by_param(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """If len is a parameter name, don't desugar."""
    src = textwrap.dedent("""
        def f(len, x):
            return len(x)
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__len__" not in attrs.get("x", set())


def test_desugar_await(map_variables, track_attributes):
    if not track_attributes: pytest.skip("requires attribute tracking")
    """await x → __await__ on x."""
    src = textwrap.dedent("""
        async def f(x):
            return await x
        """)
    m = map_variables(src)
    attrs = get_accessed_attributes(m, "f")
    assert "__await__" in attrs.get("x", set())
