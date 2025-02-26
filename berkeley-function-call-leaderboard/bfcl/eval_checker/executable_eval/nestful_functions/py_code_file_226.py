from typing import List, Tuple



def find_preferred_function(signatures: List[Tuple[str, int, List[str]]]) -> str:

    """Finds the name of the preferred function from a list of function signatures.



    The preference is based on three criteria:

    - The function is named `preferred_function`

    - The function's second argument is named `alpha`

    - The function has at least two arguments



    Args:

        signatures: A list of function signatures, where each signature is a tuple containing the function name, number of arguments, and argument names.



    Returns:

        The name of the preferred function, or None if no preferred function is found.

    """

    for signature in signatures:

        function_name, num_args, arg_names = signature

        if function_name == "preferred_function" and "alpha" in arg_names and len(arg_names) >= 2:

            return function_name

    return None

