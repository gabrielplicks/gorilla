from typing import Dict



def parse_metadata_string(metadata_string: str) -> Dict[str, str]:

    """Parses a metadata string and converts it into a dictionary.

    Args:

        metadata_string: A string containing metadata pairs of the form `key:value`.

    Returns:

        A dictionary where the keys are the keys and the values are the converted values.

    """

    metadata_dict = {}

    pairs = metadata_string.split(",")

    for pair in pairs:

        key, sep, value = pair.partition(":")

        key = key.strip()

        value = value.strip()

        if value.isdigit():

            value = int(value)

        metadata_dict[key] = value

    return metadata_dict

