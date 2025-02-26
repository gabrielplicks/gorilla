from typing import List



def create_urls(strings: List[str], base_url: str) -> List[str]:

    """Creates a list of URLs by concatenating the strings to the base URL.



    Args:

        strings: A list of strings to be concatenated to the base URL.

        base_url: The base URL to which the strings will be concatenated.



    Returns:

        A list of URLs created by concatenating the strings to the base URL.

    """

    return [base_url + '/' + s for s in strings]

