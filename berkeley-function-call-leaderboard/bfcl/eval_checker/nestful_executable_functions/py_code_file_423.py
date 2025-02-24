from typing import Tuple



BASE_URL: str = "https://example.com/api/search"



def construct_query_string(search_term: str, api_key: str) -> str:

    """Constructs a query string for the search API of a web service.

    Args:

        search_term: The search term to be included in the query string.

        api_key: The API key to be included in the query string.

    """

    query_string: str = BASE_URL + "?q=" + search_term + "&api_key=" + api_key

    return query_string

