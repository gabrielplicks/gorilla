from typing import Dict



def default_args(args: Dict, default_values: Dict) -> Dict:

    """

    Returns a new dictionary with default values replaced if they exist in `args`.



    Args:

        args: A dictionary containing the arguments.

        default_values: A dictionary containing the default values for each argument.

    """

    result = {}



    for key, value in default_values.items():

        if key in args:

            result[key] = args[key]

        else:

            result[key] = value



    return result

