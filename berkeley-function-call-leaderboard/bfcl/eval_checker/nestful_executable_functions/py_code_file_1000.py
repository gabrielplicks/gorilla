import re

from typing import Union



def simplify_slug(slug: Union[str, None]) -> str:

    """Converts a string into a simplified slug.



    The slug is lowercase and contains only alphanumeric characters and hyphens.

    Removes any leading or trailing whitespace from the input string and replaces

    any sequences of whitespace with a single hyphen. If the resulting slug is

    empty, returns 'slug-not-provided'.



    Args:

        slug: The input string to be converted into a slug.



    Returns:

        The simplified slug.

    """

    slug = re.sub(r"\s+", "-", slug.lower().strip())

    slug = re.sub(r"[^a-z0-9-]", "", slug)

    if not slug:

        return "slug-not-provided"



    return slug

