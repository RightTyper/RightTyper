import random

__version__ = "0.2.2"

from typing import Any


class RandomDict(dict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._update_internal_vectors()

    def __new__(cls, *args, **kwargs) -> "RandomDict":
        instance = super().__new__(cls)
        instance._keys = dict()
        instance._random_vector = []
        return instance

    def _update_internal_vectors(self) -> None:
        """Helper method to update _random_vector and _keys."""
        self._random_vector = list(self.keys())  # Rebuild the list of keys
        self._keys = {key: idx for idx, key in enumerate(self._random_vector)}

    def update(self, *args, **kwargs) -> None:
        """Update the dictionary and ensure _random_vector is in sync."""
        super().update(*args, **kwargs)
        self._update_internal_vectors()

    def setdefault(self, key, default=None) -> Any:
        """Override setdefault to ensure consistency of _random_vector."""
        if key not in self:
            self[key] = default
        return self[key]

    def copy(self) -> "RandomDict":
        """Return a shallow copy of the RandomDict"""
        new_rd = RandomDict(super().copy())
        new_rd._keys = self._keys.copy()
        new_rd._random_vector = self._random_vector[:]
        return new_rd

    @classmethod
    def fromkeys(cls, keys, value=None) -> "RandomDict":
        """Create a RandomDict from an iterable of keys, all mapped to the same value."""
        rd = cls()
        for key in keys:
            rd[key] = value
        rd._update_internal_vectors()  # Make sure _random_vector is populated
        return rd

    def __setitem__(self, key, value) -> None:
        """Insert or update a key-value pair"""
        super().__setitem__(key, value)
        i = self._keys.get(key, -1)

        if i == -1:
            # Add new key
            self._random_vector.append(key)
            self._keys[key] = len(self._random_vector) - 1

    def __delitem__(self, key) -> None:
        """Delete item by swapping with the last element in the random vector"""
        if key not in self._keys:
            raise KeyError(key)

        # Get the index of the item to delete
        i = self._keys[key]

        # Remove the last item from the random vector
        move_key = self._random_vector.pop()

        # Only swap if we are not deleting the last item
        if len(self._random_vector) > i:
            # Move the last item into the location of the deleted item
            self._random_vector[i] = move_key
            self._keys[move_key] = i

        del self._keys[key]
        super().__delitem__(key)

    def random_key(self) -> Any:
        """Return a random key from this dictionary in O(1) time"""
        if len(self._random_vector) == 0:
            print(
                f"Debug: _random_vector is empty. Current dict: {self}"
            )  # Debug print statement
            raise KeyError("RandomDict is empty")
        return random.choice(self._random_vector)

    def random_value(self) -> Any:
        """Return a random value from this dictionary in O(1) time"""
        return self[self.random_key()]

    def random_item(self) -> tuple[Any, Any]:
        """Return a random key-value pair from this dictionary in O(1) time"""
        k = self.random_key()
        return k, self[k]


def replace_dicts():
    # Replace dict with RandomDict
    import builtins

    builtins.dict = RandomDict  # type: ignore

    # Replace defaultdict with RandomDict

    # stash the original import for use in a custom importer
    _original_import = builtins.__import__

    def _custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Intercept imports of defaultdict to route to RandomDict"""
        module = _original_import(name, globals, locals, fromlist, level)
        if name == "collections" or (fromlist and "defaultdict" in fromlist):
            module.__dict__["defaultdict"] = RandomDict
        return module

    # Monkey-patch __import__
    builtins.__import__ = _custom_import
