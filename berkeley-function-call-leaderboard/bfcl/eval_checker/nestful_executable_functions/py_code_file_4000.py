from typing import Any



def create_list_of_duplicates(item: Any, n: int) -> list:

    """Creates a list of duplicates of an object n times using list comprehension.

    Args:

        item: The object to be duplicated.

        n: The number of times to duplicate the object.

    """

    return [item] * n

