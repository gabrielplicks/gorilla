from typing import Dict



def dict_to_csv(dictionary: Dict[str, int]) -> str:

    """Converts a dictionary to a CSV-formatted string.



    Args:

        dictionary: The dictionary to convert.



    Returns:

        A CSV-formatted string where each line represents a key-value pair in the dictionary.

    """

    lines = []

    for key, value in dictionary.items():

        lines.append(f"{key},{value}")

    return "\n".join(lines)

