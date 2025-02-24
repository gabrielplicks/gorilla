from typing import Union



def get_variable_type(value: Union[str, int, float, complex]) -> str:

    """Determines the variable type of the input value.



    Args:

        value: The input value to determine its type.



    Returns:

        The variable type as a string.

    """

    if type(value) is str:

        return "string"

    elif type(value) is int:

        return "int"

    elif type(value) is float:

        return "float"

    elif type(value) is complex:

        return "complex"

    else:

        return "other"

