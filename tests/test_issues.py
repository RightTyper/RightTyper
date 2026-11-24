import textwrap
import subprocess
import sys
from pathlib import Path


def test_issue_22(tmp_path, monkeypatch):
    t = textwrap.dedent("""\
        def extracted_function(A):
            return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(A[i] >=
                A[i + 1] for i in range(len(A) - 1))

        def optimized(A):
            return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) and all(A[i] >=
                A[i + 1] for i in range(len(A) - 1))

        def main():
            assert extracted_function([6, 5, 4, 4]) == True
            assert extracted_function([1, 2, 2, 3]) == True
            assert extracted_function([1, 3, 2]) == False

        if __name__ == '__main__':
            main()
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', 'run', 't.py'])

    assert "def extracted_function(A: list[int]) -> bool" in Path("t.py").read_text()


def test_issue_199(tmp_path, monkeypatch):
    """A wrapper that supplies an argument its caller never passed must not
    put a None inside a CallTrace.

    _get_arg_types returns None for a parameter absent from the locals mapping.
    That never happens for a real PY_START frame, but the synthetic ArgInfo built
    for wrapped-function propagation uses bind_partial, which leaves unpassed
    parameters unbound. The None used to survive into the CallTrace and crash
    whichever type transformer ran first in finish_recording -- after the traced
    run had already finished, so righttyper exited 0 having written nothing.
    """
    t = textwrap.dedent("""\
        import functools

        def deco(f):
            @functools.wraps(f)
            def wrapper(a):
                return f(a, 99)     # `b` comes from here, not from the caller
            return wrapper

        @deco
        def target(a, b):
            return a + b

        print(target(1))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run(
        [sys.executable, '-m', 'righttyper', 'run', '--only-collect', 't.py'],
        capture_output=True, text=True,
    )

    # The failure was silent: exit 0, nothing on the console, no .rt file, and
    # the traceback only in righttyper.log. Assert on the artifact, not the code,
    # which is why the CompletedProcess is deliberately not inspected.
    assert list(Path().glob("*.rt")), (
        "no .rt written; righttyper.log says:\n"
        + (Path("righttyper.log").read_text() if Path("righttyper.log").exists() else "(no log)")
    )
    log = Path("righttyper.log")
    assert not log.exists() or "exception after target execution" not in log.read_text()


def test_issue_199_synthetic_arg_does_not_erase_observations(tmp_path, monkeypatch):
    """The filler for an unbound synthetic parameter must not outrank real data.

    The synthetic ArgInfo leaves `b` unbound, so PendingCallTrace fills that slot.
    Filling it with UnknownTypeInfo -- i.e. Any -- made that trace *subsume* every
    genuine observation of `b`, because Any absorbs a union rather than vanishing
    from it, and the parameter came out unannotated despite being observed as int
    on every call. Never is the union identity, so it drops out instead.
    """
    t = textwrap.dedent("""\
        import functools

        def deco(f):
            @functools.wraps(f)
            def wrapper(a):
                return f(a, 99)
            return wrapper

        def target(a, b):
            return a + b

        print(deco(target)(1))
        print(target(2, 3))
        print(target(4, 5))
        """)

    monkeypatch.chdir(tmp_path)
    Path("t.py").write_text(t)

    subprocess.run([sys.executable, '-m', 'righttyper', 'run', '--root', '.', 't.py'])

    annotated = Path("t.py").read_text()
    sig = next(
        line for line in annotated.splitlines()
        if line.strip().startswith("def target")
    )
    assert sig.strip() == "def target(a: int, b: int) -> int:", (
        f"{sig!r}\n--- full file ---\n{annotated}"
    )


def test_issue_199_sole_synthetic_arg_is_not_annotated(tmp_path, monkeypatch):
    """The filler must not be the annotation when it is the only observation.

    A wrapper that returns without calling through still completes a synthetic
    trace for the wrapped function, so an unbound required parameter can have the
    filler as its *only* observation. `Never` is the union identity, which is what
    makes it right for merging -- but a lone `Never` survives from_set() and
    _merge_set() untouched, and `Never` is uninhabited: annotating an unobserved
    parameter with it rejects every caller.

    Under the default --no-use-typing-never this is masked, because NeverSayNeverT
    renames Never to Any and the transformer drops Any annotations. Assert both
    modes, since only one of them was ever exercised.
    """
    t = textwrap.dedent("""\
        import functools

        def deco(f):
            @functools.wraps(f)
            def wrapper(a):
                return 0        # never calls through
            return wrapper

        # b is required, so bind_partial()/apply_defaults() cannot fill it
        @deco
        def target(a, b):
            return a + b

        print(target(1))
        """)

    monkeypatch.chdir(tmp_path)

    for opt in ('--use-typing-never', '--no-use-typing-never'):
        Path("t.py").write_text(t)

        subprocess.run([sys.executable, '-m', 'righttyper', 'run', '--root', '.', opt, 't.py'])

        annotated = Path("t.py").read_text()
        sig = next(
            line for line in annotated.splitlines()
            if line.strip().startswith("def target")
        )
        # `a` is bound by the synthetic trace and legitimately annotated; `b`
        # was never observed at all, so it must be left alone.
        assert sig.strip() == "def target(a: int, b) -> int:", (
            f"{opt}: {sig!r}\n--- full file ---\n{annotated}"
        )
