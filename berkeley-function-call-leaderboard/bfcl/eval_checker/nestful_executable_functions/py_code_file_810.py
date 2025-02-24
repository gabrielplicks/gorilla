from typing import Dict, List



def filter_params(params: Dict[str, str], allowed_params: List[str]) -> Dict[str, str]:

    """

    Filters a dictionary of parameters based on a list of allowed parameters.

    If the value of any parameter is None or an empty string, it is omitted from the returned dictionary.



    Args:

        params: The dictionary of parameters.

        allowed_params: The list of allowed parameters.

    """

    filtered_params = {}

    for param in allowed_params:

        if param in params and params[param] is not None and params[param] != '':

            filtered_params[param] = params[param]

    return filtered_params

