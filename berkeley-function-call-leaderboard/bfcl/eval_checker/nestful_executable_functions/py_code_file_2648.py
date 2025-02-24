from typing import List, Tuple



def format_headers(headers: List[Tuple[str, str]]) -> List[str]:

    """Formats a list of HTTP headers, represented as a list of tuples.



    Each tuple should have the format `(key, value)` and should be converted to a string of the form `key: value`,

    where the key and value are separated by a colon.



    Args:

        headers: A list of tuples, where each tuple has exactly two elements.



    Returns:

        A list of formatted headers.

    """

    return [key + ': ' + value for key, value in headers]

