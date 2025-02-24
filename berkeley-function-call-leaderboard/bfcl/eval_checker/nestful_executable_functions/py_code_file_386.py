from typing import List



def fetch_page(items: List[str], page: int, page_size: int = 20) -> List[str]:

    """

    Fetches a subset of items from a list based on the given page number.



    Args:

        items: The list of items to paginate.

        page: The page number to fetch.

        page_size: The number of items per page (default is 20).



    Returns:

        A subset of items corresponding to the given page.

    """

    start_index = (page - 1) * page_size

    end_index = start_index + page_size

    return items[slice(start_index, end_index)]

