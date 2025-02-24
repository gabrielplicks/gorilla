from typing import List



def format_book_list(book_titles: List[str]) -> str:

    """Formats a list of book titles as a comma-separated list with "and" before the last item.

    Args:

        book_titles: A list of book titles.

    """

    if not book_titles:

        return ""

    if len(book_titles) == 1:

        return book_titles[0]

    last_item = book_titles.pop()

    formatted_list = ", ".join(book_titles)

    return f"{formatted_list} and {last_item}"

