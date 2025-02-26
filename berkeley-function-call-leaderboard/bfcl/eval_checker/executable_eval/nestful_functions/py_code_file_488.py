from typing import Union



def get_url_from_domain(domain: str, scheme: str = 'http') -> str:

    """

    Builds and returns a complete URL by composing the scheme and domain.

    The function normalizes the domain name to ensure it is in a canonical form.

    It handles cases where the domain name is already a complete URL with a scheme or without a scheme,

    and returns the original URL.



    Args:

        domain: The domain name.

        scheme: The scheme of the URL (defaults to HTTP).

    """

    if scheme in domain:

        scheme, domain = domain.split('://')

    else:

        scheme = 'http'

    domain = domain.strip('/')

    return f'{scheme}://{domain}'

