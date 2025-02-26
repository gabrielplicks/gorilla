from typing import Dict



def get_accessed_fields(dict_data: Dict[str, any]) -> Dict[str, any]:

    """Returns a subset of the dictionary consisting of only the key-value pairs for which the key is a string containing a dot.



    Args:

        dict_data: The input dictionary.

    """

    accessed_fields = {}

    for key in dict_data.keys():

        if '.' in key:

            accessed_fields[key] = dict_data[key]

    return accessed_fields

