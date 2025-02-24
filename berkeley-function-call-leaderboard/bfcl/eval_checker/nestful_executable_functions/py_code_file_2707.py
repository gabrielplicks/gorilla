import re

from typing import Dict



def format_with_placeholders(text: str, value_dict: Dict[str, str]) -> str:

    """Formats a string with placeholders using a dictionary of values.



    Args:

        text: The string with placeholders.

        value_dict: A dictionary that maps placeholder names to values.



    Returns:

        A string with the placeholders replaced by the corresponding values in the `value_dict` dictionary.

    """

    placeholders = re.findall(r'\{(.*?)\}', text)

    return text.format(**{placeholder: value_dict.get(placeholder) for placeholder in placeholders})

