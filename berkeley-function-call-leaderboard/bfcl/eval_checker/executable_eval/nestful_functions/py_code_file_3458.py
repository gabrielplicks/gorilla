import re



def convert_to_url_slug(text: str) -> str:

    """Converts a string to a URL slug by replacing spaces with hyphens and removing any characters that are not valid in a URL slug.



    Args:

        text: The input string to convert to a URL slug.



    Returns:

        The input string converted to a URL slug.

    """

    slug = text.replace(' ', '-')

    slug = re.sub(r'[^A-Za-z0-9\-]+', '', slug)

    return slug

