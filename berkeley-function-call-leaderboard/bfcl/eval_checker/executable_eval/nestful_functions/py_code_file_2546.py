from typing import Union



def is_valid_csv_url(url: Union[str, bytes]) -> bool:

    """Checks if a given URL is a valid CSV file.



    Args:

        url: The URL to check.



    Returns:

        True if the URL is valid, and False otherwise.

    """

    return isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")) and url.endswith(".csv")

