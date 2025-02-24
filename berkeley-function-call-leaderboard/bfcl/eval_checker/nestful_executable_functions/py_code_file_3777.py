import sys

from typing import Dict, List, Any



def get_dict_values_as_list(d: Dict[Any, Any]) -> List[Any]:

    """

    Returns a list of the dictionary's values.

    If the Python version is 3.9 or higher, the function uses the .values() method to get the values of the dictionary

    and converts it to a list using list(). Otherwise, it uses a list comprehension to loop over the dictionary keys

    and get the corresponding values.

    Args:

        d: The dictionary to get the values from.

    """

    if sys.version_info >= (3, 9):

        return list(d.values())

    else:

        return [d[key] for key in d]

