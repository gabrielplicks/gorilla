from typing import Any, Callable, Coroutine



import inspect



def is_coroutine_function(obj: Any) -> bool:

    """Checks if the given object is a coroutine function.



    Args:

        obj: The object to check.



    Returns:

        True if the object is a coroutine function, otherwise False.

    """

    return inspect.iscoroutinefunction(obj)

