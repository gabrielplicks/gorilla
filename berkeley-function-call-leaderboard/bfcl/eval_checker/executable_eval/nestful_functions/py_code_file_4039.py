from typing import Tuple



def convert_shortened_url(shortened_url: str, base_url: str) -> str:

    """Converts a shortened URL into a long URL by concatenating a given base URL with the string following the last slash in the shortened URL.



    Args:

        shortened_url: The shortened URL to be converted.

        base_url: The base URL to be used for concatenation.



    Returns:

        The long URL as a string.

    """

    shortened_path = shortened_url.rsplit('/', 1)[-1]

    long_url = base_url + shortened_path

    return long_url

