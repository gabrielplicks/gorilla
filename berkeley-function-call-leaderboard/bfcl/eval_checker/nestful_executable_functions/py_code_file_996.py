import inspect

from typing import Any



def is_class_method(arg: Any) -> bool:

    """Checks whether an argument is a class method.



    Args:

        arg: The argument to check.



    Returns:

        True if the argument is a class method, False otherwise.

    """

    return inspect.ismethod(arg) and inspect.isclass(arg.__self__)

