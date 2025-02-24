from typing import List, Dict, Any



def get_list_of_dictionaries(list_of_dicts: List[Dict[str, Any]]) -> List[List[Any]]:

    """Returns a list of all the values in the input list of dictionaries, provided that the value is a list itself.

    If any of the dictionaries are missing the list, the function returns None instead of raising an error.



    Args:

        list_of_dicts: A list of dictionaries.



    Returns:

        A list of lists.

    """

    output_list = []

    for dictionary in list_of_dicts:

        if "list" in dictionary:  # Check if the dictionary contains the list

            output_list.append(dictionary["list"])

        else:

            output_list.append(None)

    return output_list

