import re

from typing import Union



def custom_type_conversion(string: str, desired_type: Union[type, str]) -> Union[int, float]:

    """

    Converts a string to a specific type, such as an integer or a float.

    Args:

        string: The input string to be converted.

        desired_type: The desired type to convert the string to.

    """

    if desired_type == int:

        return int(re.sub(r'[^\d]+', '', string))

    elif desired_type == float:

        return float(re.sub(r'[^\d.]+', '', string))

    else:

        raise ValueError("Unsupported type.")

