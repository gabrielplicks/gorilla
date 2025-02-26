from typing import Union



def type_checker(arg: Union[bool, int, str, None]) -> str:

    """Determines the type of the input argument.



    Args:

        arg: The input argument to check the type of.



    Returns:

        A string indicating the type of the input argument.

    """

    if arg is None:

        return "None"

    elif arg is True or arg is False:

        return "boolean"

    elif isinstance(arg, int):

        return "integer"

    elif isinstance(arg, str):

        return "string"

    else:

        return "unknown"

