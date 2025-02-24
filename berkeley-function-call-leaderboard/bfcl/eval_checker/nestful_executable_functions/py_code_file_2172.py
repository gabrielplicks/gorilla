from typing import Union, List



def is_union_type(arg: Union[int, str, List[Union[int, str]]]) -> bool:

    """Determines if the argument is a valid union type.



    Args:

        arg: The argument to check.



    Returns:

        True if the argument is a valid union type, False otherwise.

    """

    if arg is None:

        return False

    elif isinstance(arg, int):

        return True

    elif isinstance(arg, str):

        return True

    elif isinstance(arg, list):

        for element in arg:

            if not is_union_type(element):

                return False

        return True

    else:

        return False

