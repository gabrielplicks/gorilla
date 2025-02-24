from typing import Union



def parse_parameter_and_sum(parameter_string: str) -> Union[int, None]:

    """Parses a string containing a parameter name and value, and returns the sum of all digits in the value.

    Returns None if the parameter name doesn't match or the value is empty.

    Args:

        parameter_string: A string containing a parameter name and value, separated by '='.

    """

    parameter_name, parameter_value = parameter_string.split('=')

    if parameter_name != 'x':

        return None

    if not parameter_value:

        return None

    return sum(map(int, parameter_value))

