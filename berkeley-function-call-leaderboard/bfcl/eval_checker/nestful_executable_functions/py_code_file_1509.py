from typing import Dict



def sorted_dict_string(d: Dict[str, int]) -> str:

    """Returns a string of the dictionary's key-value pairs, separated by commas and sorted in ascending order of the values.

    Each key-value pair is represented as a string in the format `{key}: {value}`.

    Args:

        d: A dictionary of strings and numbers.

    """

    sorted_pairs = sorted(d.items(), key=lambda x: x[1])

    strings = []

    for key, value in sorted_pairs:

        strings.append(f"{key}: {value}")

    return ", ".join(strings)

