from typing import List



class Field:

    def __init__(self, name: str, value: str):

        self.name = name

        self.value = value



def format_field_values(fields: List[Field]) -> str:

    """Formats a list of field objects into a single string containing the field values separated by commas, where field values containing commas are first enclosed in double quotes.



    Args:

        fields: A list of field objects, each with a `.value` and `.name` attribute.



    Returns:

        A string containing the formatted field values.

    """

    formatted_values = []

    for field in fields:

        value = field.value

        if "," in value:

            value = f'"{value}"'

        formatted_values.append(value)

    return ",".join(formatted_values)

