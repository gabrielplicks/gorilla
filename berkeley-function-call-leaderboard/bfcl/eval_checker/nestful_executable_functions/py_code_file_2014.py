from typing import List, Dict, Union



def extract_from_list_of_dicts(input: List[Union[Dict, List[Dict]]], field_name: str) -> List:

    """Extracts a certain field from each dictionary in a list of dictionaries.



    Args:

        input: A list of dictionaries, each dictionary containing the required field, or a list of dictionaries, each dictionary containing a list of the required field.

        field_name: The name of the field to extract.



    Returns:

        A list of the extracted values.

    """

    if isinstance(input[0], dict):

        return [d[field_name] for d in input]

    elif isinstance(input[0], list):

        return [d[0][field_name] for d in input]

    else:

        raise TypeError("Input must be a list of dictionaries.")

