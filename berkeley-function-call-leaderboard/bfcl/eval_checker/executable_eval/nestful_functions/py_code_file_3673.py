from typing import Any, Tuple

import inspect



def get_params_for_method(method: Any) -> list[Tuple[str, Any]]:

    """

    Returns a list of tuples containing the parameter name and its type hint, if one exists.



    Args:

        method: The method to inspect.



    Returns:

        A list of tuples containing the parameter name and its type hint, if one exists.

    """

    params = inspect.signature(method).parameters

    param_list = []



    for param_name, param in params.items():

        type_hint = param.annotation

        param_list.append((param_name, type_hint if type_hint != inspect.Parameter.empty else None))



    return param_list

