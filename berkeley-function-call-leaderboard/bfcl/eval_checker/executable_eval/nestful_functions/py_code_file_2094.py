from typing import Dict

import inspect



def create_signature(params: Dict[str, object]) -> inspect.Signature:

    """Creates a signature object from a dictionary of parameters.



    Args:

        params: A dictionary containing the parameters for the signature.

    """

    parameters = [inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=value) for key, value in params.items()]

    return inspect.Signature(parameters)

