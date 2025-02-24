from typing import Union



def convert_string_to_boolean(string: str) -> Union[bool, None]:

    """Converts a boolean string representation to a boolean.



    Args:

        string: A boolean string representation.



    Returns:

        A boolean if the string is 'True' or 'False', None otherwise.

    """

    if string == 'True':

        return True

    elif string == 'False':

        return False

    else:

        return None

