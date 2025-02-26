import urllib.request



def get_url_content(url: str) -> str:

    """Gets the text content of a URL's page.



    Args:

        url: The URL to get the text content from.



    Returns:

        The text content of the URL's page.



    Raises:

        Exception: If the HTTP request is not successful.

    """

    with urllib.request.urlopen(url) as response:

        if response.getcode() == 200:

            return response.read().decode()

        else:

            raise Exception("HTTP request failed with status code", response.getcode())

