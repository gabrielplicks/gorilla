from typing import Union



def increment_value(x: Union[int, float, complex, str, bool]) -> Union[int, float, complex, str, bool]:

    """

    Increment the value of an object based on its type.



    Args:

        x: The object to increment.



    Returns:

        The incremented value of the object.



    Raises:

        ValueError: If the type of the object is not supported.

    """

    x_type = type(x)

    if x_type in (int, float, complex):

        return x + 1

    elif x_type is str:

        return x + 'a'

    elif x_type is bool:

        return x or True

    else:

        raise ValueError('Invalid type')

