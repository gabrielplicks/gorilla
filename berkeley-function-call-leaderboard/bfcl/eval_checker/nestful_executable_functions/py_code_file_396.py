import re

from typing import List



def find_words_with_keyword(words: List[str], keyword: str, case_sensitive: bool = False) -> List[str]:

    """

    Filters a list of strings based on the presence of a specified keyword.

    Returns a new list that contains only the words that contain the keyword.

    Uses keyword arguments to specify case sensitivity, and defaults it to False

    to make the matching case-insensitive.



    Args:

        words: The list of strings to filter.

        keyword: The keyword to search for in the strings.

        case_sensitive: Whether the search should be case-sensitive. Defaults to False.

    """

    pattern = re.compile(keyword, flags=re.IGNORECASE if not case_sensitive else 0)

    return [word for word in words if pattern.search(word)]

