from typing import Union



def map_boolean_to_binary_representation(boolean_value: Union[int, bool]) -> bool:

    """Maps a boolean value to a value based on the binary representation of the boolean.



    Args:

        boolean_value: The boolean value to be mapped.



    Returns:

        The mapped value.

    """

    return [False, True][boolean_value]

