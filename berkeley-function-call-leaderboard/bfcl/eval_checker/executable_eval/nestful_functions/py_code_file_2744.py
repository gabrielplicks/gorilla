from typing import Dict, List



def compare_api(api_1: Dict[str, List[str]], api_2: Dict[str, List[str]]) -> Dict[str, List[str]]:

    """

    Compares two dictionaries representing API endpoints and returns a new dictionary containing the differences.

    Args:

        api_1: The first API dictionary.

        api_2: The second API dictionary.

    """

    result = {}

    for key, value in api_1.items():

        if key not in api_2:

            result[key] = value

        else:

            diff = list(set(api_2[key]) - set(value))

            if diff:

                result[key] = diff

    return result

