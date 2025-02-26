from typing import Dict, Any



def get_functions_in_module(module: Any) -> Dict[str, Any]:

    """Returns a dictionary of the names of all functions in the input module, indexed by the names of these functions.



    Args:

        module: The input module.



    Returns:

        A dictionary of the names of all functions in the input module, indexed by the names of these functions.

    """

    functions = {}

    for name in dir(module):

        obj = getattr(module, name)

        if callable(obj):

            functions[name] = obj

    return functions

