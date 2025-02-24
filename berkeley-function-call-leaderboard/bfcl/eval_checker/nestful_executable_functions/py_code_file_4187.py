import re

from typing import List



def search_terms(text: str, terms: List[str]) -> int:

    """Counts the number of search terms that appear in the text string.



    Args:

        text: The text string to search.

        terms: A collection of search terms.



    Returns:

        The number of search terms that appear in the text string.

    """

    return sum(

        1 for term in terms if (

            term in text or

            re.search(term, text) or

            re.search(r"\b{}\b".format(term), text)

        )

    )

