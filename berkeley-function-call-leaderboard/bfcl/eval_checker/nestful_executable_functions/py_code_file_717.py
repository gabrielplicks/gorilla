from typing import Dict



def transform_vals(data: Dict[str, str], template: str) -> Dict[str, str]:

    """Transforms the values of a dictionary according to a template string.



    The template string is a format string that may contain references to keys in the dictionary.

    The function returns a new dictionary with transformed values.



    Args:

        data: A dictionary of keys and values.

        template: A format string.



    Returns:

        A new dictionary with transformed values.

    """

    transformed_val = template.format(**data)

    return {**data, 'transformed_val': transformed_val}

