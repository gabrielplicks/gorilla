from typing import List



def generate_method_name(params: List[str]) -> str:

    """Generates a method name from a set of input parameters.

    The method name is a concatenation of the input parameter names, separated by underscores.

    Args:

        params: A list of input parameter names.

    """

    return '_'.join(params)

