from typing import Dict



def transform_dict_to_string(dict: Dict[str, str]) -> str:

    """Transforms a dictionary of key-value pairs into a single string containing all the keys and values separated by commas and colons (no spaces), where the values are formatted as strings if the values are not numbers.



    Args:

        dict: The dictionary to transform.



    Returns:

        The transformed string.

    """

    result = []

    for key, value in dict.items():

        if not isinstance(value, (int, float)):

            value = str(value)  # Convert to string if not a number

        result.append(f"{key}:{value}")

    return ",".join(result)

