from typing import Dict



def call_foo_with_params(params: Dict[str, any]) -> any:

    """Calls a function `foo` by unpacking a dictionary `params` with named parameters.



    Args:

        params: A dictionary with named parameters.

    """

    return foo(**params)

