from typing import Callable

import inspect



def construct_function_name(function_name: str) -> str:

    """Constructs a string representing the full function name and its arguments.



    Args:

        function_name: The name of the function.



    Returns:

        A string representing the full function name and its arguments.

    """

    fn: Callable = globals().get(function_name)

    arguments: tuple = inspect.getfullargspec(fn).args

    return f"{function_name}({', '.join(arguments)})"

