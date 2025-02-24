import inspect

from typing import Any, Type



def is_valid_class(cls: Any) -> bool:

    """Checks if the given object is a valid class.



    A class is considered valid if it has a `__name__` attribute and its name matches the name of the class itself.

    If the input object is not a class, the function returns `False`.



    Args:

        cls: The object to check.



    Returns:

        A boolean value indicating whether the class is valid or not.

    """

    if not inspect.isclass(cls):

        return False

    if not hasattr(cls, '__name__'):

        return False

    return cls.__name__ == cls.__name__

