from typing import Dict, Any



def extract_function_names(dictionary: Dict[str, Any]) -> list[str]:

    """Extracts the names of functions from a nested dictionary.



    Args:

        dictionary: The nested dictionary to traverse.



    Returns:

        A list of function names.

    """

    function_names = []

    def traverse(d: Dict[str, Any]):

        for key, value in d.items():

            if isinstance(value, dict):

                traverse(value)

            elif callable(value):

                function_names.append(key)

    traverse(dictionary)

    return function_names

