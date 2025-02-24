from typing import Any



def type_and_value(value: Any) -> str:

    """

    Returns a string representation of the input value and its static type.



    Args:

        value: The input value to be displayed. Can be any type.

    """

    return f"{type(value).__name__}: {value}"

