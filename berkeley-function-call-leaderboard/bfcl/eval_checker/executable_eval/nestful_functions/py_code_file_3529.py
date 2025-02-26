import inspect



def get_meta_data(func_str: str) -> inspect.FullArgSpec:

    """Gets the metadata of a function from its string representation.



    Args:

        func_str: The string representation of the function.



    Returns:

        The metadata of the function.

    """

    func = eval(func_str)

    return inspect.getfullargspec(func)

