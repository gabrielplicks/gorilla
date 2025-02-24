from typing import Union



def enclose(input_string: Union[str, None]) -> str:

    """Encloses an input string in either single or double quotes based on its content.



    Args:

        input_string: The input string to be enclosed.



    Returns:

        The input string enclosed in either single or double quotes.

    """

    if "'" in input_string:

        return f'"{input_string}"'

    elif '"' in input_string:

        return f"'{input_string}'"

    elif "'" in input_string and '"' in input_string:

        return f'"{input_string}"'

    else:

        return f'"{input_string}"'

