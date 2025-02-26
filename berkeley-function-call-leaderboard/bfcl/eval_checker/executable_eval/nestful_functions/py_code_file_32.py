from typing import Dict



def average_string_length(dictionary: Dict[str, list[str]]) -> Dict[str, float]:

    """Calculates the average length of strings in the corresponding list of strings for each key in the dictionary.



    Args:

        dictionary: A dictionary of strings to lists of strings.



    Returns:

        A new dictionary with the original keys and their average string length as values.

    """

    result = {}

    for key, values in dictionary.items():

        total_length = 0

        for value in values:

            total_length += len(value)

        result[key] = total_length / len(values)

    return result

