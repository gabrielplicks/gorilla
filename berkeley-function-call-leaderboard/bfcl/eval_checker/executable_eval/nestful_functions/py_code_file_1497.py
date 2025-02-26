from typing import Dict, Union



def get_constant_value(string: str) -> Union[float, None]:

    """Returns the constant value corresponding to the given string.



    Args:

        string: The string representing the constant name.



    Returns:

        The constant value if the string is a recognized constant name, or `None` otherwise.

    """

    constants: Dict[str, float] = {

        'pi': 3.14,

        'e': 2.72,

        'phi': 1.62,

        'c': 299792458,

        'G': 6.6743e-11

    }

    if string not in constants:

        return None



    return constants[string]

