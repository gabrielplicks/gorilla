from typing import List, Optional



def get_book_authors(book: dict) -> List[str]:

    """Returns a list of authors for a given book object.



    Args:

        book: A dictionary representing a book.



    Returns:

        A list of authors. If the book has a `primary_author` attribute,

        return a list with a single element containing the primary author.

        Otherwise, if the book has an `authors` attribute, return a list

        of all authors. If neither of these attributes exist, return an

        empty list.

    """

    if hasattr(book, "primary_author"):

        return [book.primary_author]

    elif hasattr(book, "authors"):

        return book.authors

    else:

        return []

