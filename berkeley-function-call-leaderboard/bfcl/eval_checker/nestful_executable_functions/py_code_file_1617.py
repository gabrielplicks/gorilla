from typing import Any



def count_elements_in_object(obj: Any) -> int:

    """Counts the total number of elements in an object that supports the `__getitem__` and `__len__` protocols.

    Args:

        obj: The object to count the elements in.

    """

    if hasattr(obj, "__len__"):

        return len(obj)

    elif hasattr(obj, "__getitem__"):

        total = 0

        index = 0

        while True:

            try:

                obj[index]

                total += 1

                index += 1

            except IndexError:

                break

        return total

    else:

        raise TypeError("Object does not support __getitem__ and __len__ protocols.")

