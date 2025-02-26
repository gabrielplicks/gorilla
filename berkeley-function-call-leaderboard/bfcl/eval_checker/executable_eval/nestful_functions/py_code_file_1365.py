from typing import Any



def has_search_function(obj: Any) -> bool:

    """Checks if the given object has a function called `search` that takes two arguments (encoding and value).

    Returns `True` if the call to `search` returns `True`, otherwise it returns `False`.



    Args:

        obj: The object to check.

    """

    if not hasattr(obj, 'search') or not callable(getattr(obj, 'search')):

        return False



    result = obj.search('encoding', 'value')

    return result == True

