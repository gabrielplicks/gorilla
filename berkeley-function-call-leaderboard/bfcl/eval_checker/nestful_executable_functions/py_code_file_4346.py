from typing import List, Any



def collect_params(*args: Any, reverse: bool = False) -> List[Any]:

    """Collects an arbitrary number of parameters and returns a list of parameter values.

    Args:

        args: The arbitrary number of parameters.

        reverse: A boolean flag to determine whether to reverse the returned list of parameter values.

    """

    params = list(args)

    if reverse:

        return params[::-1]

    return params

