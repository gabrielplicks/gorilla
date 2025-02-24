from typing import List, Dict



def get_error_values(list_of_dicts: List[Dict[str, str]]) -> List[str]:

    """

    Returns a list of error value strings from a list of dictionaries.



    Args:

        list_of_dicts: A list of dictionaries containing error information.



    Returns:

        A list of error value strings.

    """

    error_values = []

    for dictionary in list_of_dicts:

        error_code = dictionary.get('error_code')

        error_message = dictionary.get('error_message')

        params = dictionary.get('params')

        error_value = f'ERROR: {error_code}: {error_message}: {params}'

        error_values.append(error_value)

    return error_values

