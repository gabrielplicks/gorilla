import inspect

import types



def get_generator_co_code(func: types.FunctionType) -> bytes:

    """Determines whether the given function is a generator function.



    If it is, returns its `co_code` attribute; otherwise, returns `None`.



    Args:

        func: The function to check.

    """

    if inspect.isgeneratorfunction(func):

        return func.__code__.co_code

    return None

