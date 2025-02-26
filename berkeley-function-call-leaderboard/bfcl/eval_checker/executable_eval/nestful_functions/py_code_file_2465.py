from typing import List, Union



def url_from_parts(*parts: Union[str, List[str]]) -> str:

    """Constructs a URL from a variable number of parts.



    Args:

        parts: The parts of the URL. Each part can be a string or a list of strings.

    """

    return "/".join(parts)

