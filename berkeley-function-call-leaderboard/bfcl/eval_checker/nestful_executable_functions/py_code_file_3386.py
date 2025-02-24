from typing import Dict



def merge_two_dicts(dict1: Dict, dict2: Dict) -> Dict:

    """Merges two Python dictionaries recursively.



    Args:

        dict1: The first dictionary to merge.

        dict2: The second dictionary to merge.



    Returns:

        A new dictionary that contains the combined contents of `dict1` and `dict2`.

    """

    result = dict1.copy()



    def merge_helper(dict1: Dict, dict2: Dict):

        for key, value in dict2.items():

            if key in dict1:

                if isinstance(value, dict):

                    merge_helper(dict1[key], value)

                else:

                    dict1[key] = value

            else:

                dict1[key] = value



    merge_helper(result, dict2)

    return result

