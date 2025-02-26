from typing import List



def check_query_param(url: str) -> bool:

    """

    Checks if the parameter name "q" exists in the URL.

    Args:

        url: The URL to check for the parameter name "q".

    """

    query_part = url.split('?')[-1]  # Extract the part after the "?" character

    param_names: List[str] = [param.split('=')[0] for param in query_part.split('&')]  # Extract the parameter names

    return 'q' in param_names

