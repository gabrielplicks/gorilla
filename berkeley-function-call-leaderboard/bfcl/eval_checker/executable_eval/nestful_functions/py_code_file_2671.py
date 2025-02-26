from typing import Dict



def parse_parameters(params_string: str) -> Dict[str, str]:

    """Parses a string into a dictionary.

    Args:

        params_string: A string in the format of "key1=value1,key2=value2,key3=value3".

    Returns:

        A dictionary where the keys are the parts of the string before the `=` symbol and

        the values are the parts after the `=` symbol.

    """

    params_dict = {}

    for param in params_string.split(","):

        key, value = param.split("=")

        params_dict[key] = value

    return params_dict

