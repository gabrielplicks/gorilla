from typing import Dict



def parse_fields_and_values(input_string: str) -> Dict[str, str]:

    """Parses a string and returns a dictionary of field names and values.



    Args:

        input_string: The input string containing field names and values.



    Returns:

        A dictionary of field names and values.

    """

    lines = input_string.split('\n')

    fields_and_values: Dict[str, str] = {}



    for line in lines:

        fields = line.split(':')

        if len(fields) > 1:

            fields_and_values[fields[0]] = fields[1].strip()

        else:

            fields_and_values[line] = ""



    return fields_and_values

