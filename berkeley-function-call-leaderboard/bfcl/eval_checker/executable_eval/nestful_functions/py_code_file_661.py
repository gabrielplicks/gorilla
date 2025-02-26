from typing import Union



def parse_string_field(input_string: str, field_name: str) -> Union[str, int, None]:

    """Parses a string field and returns the value of the field if it exists.



    Args:

        input_string: The input string in the format "field_name=field_value".

        field_name: The name of the field to search for.



    Returns:

        The value of the field if it exists, or None if the field does not exist.

    """

    key_value_pairs = input_string.split()

    for pair in key_value_pairs:

        key, value = pair.split("=")

        if key == field_name:

            return value



    return None

