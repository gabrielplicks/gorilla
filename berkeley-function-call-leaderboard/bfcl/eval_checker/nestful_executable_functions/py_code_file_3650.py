import inspect

from typing import Any, Dict



def collect_args(func: callable) -> Dict[str, Any]:

    """Collects all the arguments passed to a function and returns them as a dictionary.



    Args:

        func: The function to collect arguments from.



    Returns:

        A dictionary containing all the arguments passed to the function.

    """

    sig = inspect.signature(func)

    params = sig.parameters

    arg_dict = {}



    for name, param in params.items():

        default = param.default

        kind = param.kind

        if kind == inspect.Parameter.VAR_POSITIONAL:

            arg_dict[name] = args if args else default

        elif kind == inspect.Parameter.VAR_KEYWORD:

            arg_dict[name] = kwargs if kwargs else default

        else:

            arg_dict[name] = default



    return arg_dict

