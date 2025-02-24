from typing import Any



def get_type_string(arg: Any) -> str:

    """Returns a string representation of the argument's type.



    Args:

        arg: The argument to check the type of.



    Returns:

        A string that exactly matches the string representation of the

        corresponding Python built-in type.

    """

    type_str = str(type(arg))

    type_str = type_str.replace("<class '", "")

    type_str = type_str.replace("'>", "")

    return type_str

