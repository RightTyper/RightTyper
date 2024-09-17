import random

__version__ = '0.2.2'

class RandomDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_vector = list(self.keys())  # Directly convert the keys into a list

        # Create the _keys dictionary with indexes corresponding to _random_vector
        self._keys = {key: idx for idx, key in enumerate(self._random_vector)}

        # Set last_index based on the length of _random_vector
        self.last_index = len(self._random_vector) - 1

    def copy(self):
        """ Return a shallow copy of the RandomDict """
        new_rd = RandomDict(super().copy())
        new_rd._keys = self._keys.copy()
        new_rd._random_vector = self._random_vector[:]
        new_rd.last_index = self.last_index
        return new_rd

    @classmethod
    def fromkeys(cls, keys, value=None):
        """Create a RandomDict from an iterable of keys, all mapped to the same value."""
        rd = cls()
        for key in keys:
            rd[key] = value
        return rd

    def __setitem__(self, key, value):
        """ Insert or update a key-value pair """
        super().__setitem__(key, value)
        i = self._keys.get(key, -1)

        if i == -1:
            # Add new key
            self.last_index += 1
            self._random_vector.append(key)
            self._keys[key] = self.last_index

    def __delitem__(self, key):
        """ Delete item by swapping with the last element in the random vector """
        if key not in self._keys:
            raise KeyError(key)

        # Get the index of the item to delete
        i = self._keys[key]

        # Remove the last item from the random vector
        move_key = self._random_vector.pop()

        # Only swap if we are not deleting the last item
        if i != self.last_index:
            # Move the last item into the location of the deleted item
            self._random_vector[i] = move_key
            self._keys[move_key] = i

        self.last_index -= 1
        del self._keys[key]
        super().__delitem__(key)

    def random_key(self):
        """ Return a random key from this dictionary in O(1) time """
        if len(self) == 0:
            raise KeyError("RandomDict is empty")
        i = random.randint(0, self.last_index)
        return self._random_vector[i]

    def random_value(self):
        """ Return a random value from this dictionary in O(1) time """
        return self[self.random_key()]

    def random_item(self):
        """ Return a random key-value pair from this dictionary in O(1) time """
        k = self.random_key()
        return k, self[k]


def replace_dicts():
    # Replace dict with RandomDict
    import builtins
    builtins.dict = RandomDict

    # Replace defaultdict with RandomDict

    # stash the original import for use in a custom importer
    _original_import = builtins.__import__

    def _custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Intercept imports of defaultdict to route to RandomDict"""
        module = _original_import(name, globals, locals, fromlist, level)
        if name == "collections" or (fromlist and "defaultdict" in fromlist):
            module.__dict__['defaultdict'] = RandomDict
        return module

    # Monkey-patch __import__
    builtins.__import__ = _custom_import
