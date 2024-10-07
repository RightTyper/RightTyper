from righttyper.get_import_details import *


def test_get_import_details():
    import numpy
    import numpy as np
    from rich.console import Console as ApplePie
    from rich.console import Console as Banana

    # Test the functions
    assert get_import_details(numpy.ndarray) == ImportDetails(
        "ndarray",
        frozenset({}),
        "numpy",
        frozenset({"np"}),
    )
    assert get_import_details(np.ndarray) == ImportDetails(
        "ndarray",
        frozenset({}),
        "numpy",
        frozenset({"np"}),
    )
    assert get_import_details(Banana) == ImportDetails(
        "Console",
        frozenset(), #{"ApplePie", "Banana"}),
        "rich.console",
        frozenset({}),
    )
    # print_possible_imports(get_import_details(Banana))
    assert get_import_details(ApplePie) == ImportDetails(
        "Console",
        frozenset(), #{"ApplePie", "Banana"}),
        "rich.console",
        frozenset({}),
    )
    assert get_import_details(lru_cache) == ImportDetails(
        "lru_cache",
        frozenset({}),
        "functools",
        frozenset({}),
    )
