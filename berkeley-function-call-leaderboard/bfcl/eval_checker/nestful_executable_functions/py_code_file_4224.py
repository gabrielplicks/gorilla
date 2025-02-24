import inspect



def get_source_code(func: callable) -> str:

    """Returns the source code of a Python function.



    Args:

        func: The function to retrieve the source code from.



    Returns:

        The source code of the function as a string.

    """

    return inspect.getsource(func)

