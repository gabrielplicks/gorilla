import urllib.parse



def normalize_url(url: str) -> str:

    """Normalizes a URL by removing unnecessary components and appending a trailing slash if necessary.

    Args:

        url: The URL to normalize.

    """

    parsed = urllib.parse.urlparse(url)

    path = parsed.path

    if not path.endswith('/'):

        path += '/'

    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))

