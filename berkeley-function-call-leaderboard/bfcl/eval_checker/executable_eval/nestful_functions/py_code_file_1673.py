import random

random.seed(42)
import string

from typing import Union



def shorten_link(url: str) -> str:

    """Shortens a URL string according to specific rules.



    Args:

        url: The URL string to be shortened.



    Returns:

        A shortened version of the URL string.

    """

    shortened_url = ''



    while len(shortened_url) < 8 or 'a' not in shortened_url or 'b' not in shortened_url or 'c' not in shortened_url:

        shortened_url = ''.join(random.choices(string.ascii_letters + string.digits, k=8))



    for char in url:

        if char.isalpha() and char not in shortened_url:

            while char in shortened_url:

                char = random.choice(string.ascii_letters)

            shortened_url += char

        else:

            shortened_url += char



    return shortened_url

