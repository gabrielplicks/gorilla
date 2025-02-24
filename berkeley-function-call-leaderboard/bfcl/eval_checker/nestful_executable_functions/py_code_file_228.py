from typing import Dict



def format_dict_as_tsv(data: Dict[str, str]) -> str:

    """

    Formats the given dictionary as a tab-separated values string.

    Args:

        data: A dictionary containing the data to be formatted.

    Raises:

        ValueError: If the input dictionary contains duplicate keys or values.

    """

    keys = data.keys()

    values = data.values()

    if len(data) != len(keys):

        raise ValueError("The input dictionary contains duplicate keys.")

    if len(data) != len(values):

        raise ValueError("The input dictionary contains duplicate values.")

    tsv_lines = []

    for key, value in zip(keys, values):

        tsv_lines.append(f"{key}\t{value}")

    return "\n".join(tsv_lines)

