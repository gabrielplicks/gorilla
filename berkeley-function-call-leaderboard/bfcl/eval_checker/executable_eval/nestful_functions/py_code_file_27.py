from typing import Dict



def str_format_named_params(string: str, **params: Dict[str, str]) -> str:

    """Formats a string by replacing placeholders with their corresponding values from a dictionary.



    Args:

        string: The string to format.

        params: A dictionary of placeholder names and their corresponding values.

    """

    return string.format(**params)

