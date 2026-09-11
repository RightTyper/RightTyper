# Probing objects RightTyper did not create

RightTyper inspects values and types belonging to the program being traced. Those objects
are written by someone else, and some of them do not behave the way the language's
introspection protocols suggest they will.

## The invariant

`isinstance(x, type)`, `hasattr(x, "__code__")` and `isinstance(x, abc.Hashable)` tell you
what an object **claims**, not what it will **tolerate**.

Every operation performed on a value that came from the traced program — `hash()`,
`issubclass()`, `getattr()`, following a `__wrapped__` chain — must survive an object that
lies, raises, or synthesizes a fresh attribute on every access. Real examples already met:

| Object | Claims | Actually |
|---|---|---|
| `unittest.mock` `_Call` | has `__code__` | `__getattr__` returns a child `_Call` for any name, and `_Call` is unhashable |
| a `TypedDict` subclass | `isinstance(x, type)` | refuses `issubclass` by design |
| a `Protocol` with data members | `isinstance(x, type)` | refuses `issubclass` unless `runtime_checkable` |
| a class whose metaclass defines `__eq__` and not `__hash__` | is a class | unhashable, so it cannot key a dict or join a set |
| a class with a `__hash__` that raises | `isinstance(x, abc.Hashable)` | raises when hashed — the ABC only checks `__hash__ is not None` |
| a zope-`interface`-hooked class | ordinary attributes | `getattr` runs a descriptor that raises |

## How these are found

Not by reading. Each of these was plausible-looking code that failed only when executed
against a real adversarial object, and each was found in a full test-suite run rather than
in review. When you fix one, add a regression test that **runs the path** with an object of
that shape; asserting on the helper in isolation does not catch the next one.

## The helpers

Each carries a docstring explaining its specific hazard. Read that before adding a caller —
the reasoning is in the source, not repeated here.

The last column says where a helper comes from, since most of them arrive with the fixes
still under review rather than being on `main` today.

| You need to | Use | In | Lands with |
|---|---|---|---|
| know whether an attribute exists, without running a descriptor | `_safe_getattr` | `type_id.py` | on `main` |
| know which attributes a type actually resolves | `_probed_attrs` | `generalize.py` | #196 |
| get a code object, if there really is one | `code_of` | `righttyper_types.py` | #195 |
| check something really is a code object | `has_code` | `righttyper_types.py` | on `main` |
| use a value as a dict key or set member | `is_hashable` | `righttyper_utils.py` | #198 |
| ask a subtype question that may be refused | `safe_issubclass` | `righttyper_utils.py` | #202 |
| look up a possibly-unhashable class in a dict | `_lookup_type` | `type_id.py` | #198 |
| follow a `__wrapped__` chain | `unwrap` | `righttyper_utils.py` | on `main` |
| say "this observation contributes nothing" | `NeverTypeInfo` | `typeinfo.py` | #201 |

Note the split between the first row and the third. There is no single "probe safely"
primitive, and the next section says why.

## Checklist when adding or reviewing a call site

- **`inspect.getattr_static` answers with the descriptor, not the value.** For anything held
  as a slot or getset on the type — `__code__`, `__qualname__`, `__module__`, `__func__` — a
  static read hands back the descriptor object rather than what attribute access would give.
  So it answers *"does this attribute exist?"* and never *"what is its value?"*. Reading a
  value off a traced-program object means a dynamic `getattr` inside a `try`, with the
  **result's type checked** — which is what `code_of` is. Static probing is also wrong for
  `__wrapped__` on a classmethod or staticmethod taken from a class `__dict__` (it yields a
  member descriptor) and blind to one on a bound method (it finds nothing), and `__func__`
  cannot be used to normalize past either, being a descriptor itself.
- **`issubclass(a, b)`: `__subclasscheck__` lives on `b`.** Only guard calls whose *second*
  operand comes from the traced program. The many `issubclass(x, <fixed ABC>)` calls in this
  codebase cannot raise and need nothing.
- **A cache keyed on a type object must be bounded.** The key keeps the class alive, so an
  unbounded cache pins every class ever probed — including the throwaway ones that
  `make_dataclass`, `namedtuple` and mock autospec generate — for the life of the process.
- **A cycle guard keyed on `id()` is not enough.** It cannot detect an object that returns a
  brand-new child on every access, because every child has a new id. Cap the depth as well,
  or the loop allocates until the process is killed.
- **The identity for "nothing observed here" is `Never`, not `Unknown`.** `Never` disappears
  from a union as soon as anything else joins it; `Unknown` is `Any` and subsumes the union
  instead, silently discarding real observations of the same parameter from other calls.
- **Catch narrowly where a wrong answer would be silent, broadly where it would be fatal.**
  `TypeError` is how "this class does not support that operation" is spelled, by `TypedDict`,
  by `Protocol` and by `type.__subclasscheck__` itself; swallowing anything wider in
  `safe_issubclass` produces a quietly wrong type instead of a visible fault. The probes
  reached from the `CALL` handler are the exception: it is registered process-globally, so an
  exception escaping it surfaces inside the traced program at the call instruction. There,
  "not something we can annotate" is the answer that keeps the program running.
