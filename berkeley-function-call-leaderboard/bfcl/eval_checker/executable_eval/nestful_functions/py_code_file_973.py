import inspect

from typing import Dict



def get_functions_and_docstrings(module_name: str) -> Dict[str, str]:

    """Returns a dictionary of functions and their docstrings for a given module/package.



    Args:

        module_name: The name of the module/package.



    Returns:

        A dictionary where the keys are function names and the values are their docstrings.

    """

    module = __import__(module_name)

    functions_with_docstrings = {}

    for name, obj in inspect.getmembers(module):

        if inspect.isfunction(obj):

            functions_with_docstrings[name] = obj.__doc__

    return functions_with_docstrings

