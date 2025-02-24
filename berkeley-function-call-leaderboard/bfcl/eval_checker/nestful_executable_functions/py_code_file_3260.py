from typing import Optional



def find_search_term(content: str, term: str) -> Optional[str]:

    """Searches for a search term in a longer string and returns the first occurrence.

    If there is no match, returns an empty string.

    Args:

        content: The longer string to search in.

        term: The search term to find.

    """

    index = content.find(term)

    if index == -1:

        return ""

    return content[index:]

