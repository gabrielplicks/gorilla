from typing import Dict



def concatenate_keys_and_values(dictionary: Dict[str, int]) -> list:

    """Constructs a list of concatenated keys and values from a dictionary.

    Args:

        dictionary: The dictionary to extract keys and values from.

    Returns:

        A list of strings, where each string is in the format "key:value".

    """

    concatenated_list = []

    for key, value in dictionary.items():

        concatenated_list.append(f"{key}:{value}")

    return concatenated_list

