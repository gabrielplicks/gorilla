from typing import List, Union



def validate_my_class(instance: List[Union[int, float]]) -> None:

    """Validates an instance of a generic custom class `MyClass` that accepts a list of `int`s, floats, or a combination of both.

    If the instance is not valid, raises a `ValueError` exception with a meaningful error message.



    Args:

        instance: The instance to validate.

    """

    if not isinstance(instance, list):

        raise ValueError("The instance is not a list.")



    for element in instance:

        if not isinstance(element, (int, float)):

            raise ValueError(f"The element {element} is not an int or float.")

