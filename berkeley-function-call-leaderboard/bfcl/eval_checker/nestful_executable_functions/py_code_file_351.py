from typing import List



def make_query_string(terms: List[str], start: int = 0) -> List[str]:

    """Creates a properly-formatted query string from a given list of search terms.



    Args:

        terms: A list of search terms (strings).

        start: The starting index for the list of terms (defaults to 0).



    Returns:

        A list of query strings.

    """

    query_strings = []

    while start < len(terms):

        query_string = "q=" + " ".join(terms[start:start+3])

        if len(query_string) > 2000:

            query_string += "\u200b"

        query_strings.append(query_string)

        start += 3

    return query_strings

