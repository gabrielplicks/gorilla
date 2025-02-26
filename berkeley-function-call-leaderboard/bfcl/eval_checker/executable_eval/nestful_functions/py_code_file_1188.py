from typing import Dict



def get_book_word_counts(filename: str) -> Dict[str, int]:

    """Calculates the word counts for each word in a book.



    Args:

        filename: The name of the file containing the book text.



    Returns:

        A dictionary of word counts for each word in the book.

    """

    with open(filename, 'r') as book:

        contents = book.read()

        words = contents.split()  # Tokenize text on whitespace



        word_counts = {}

        for word in words:

            word_counts[word] = word_counts.get(word, 0) + 1  # Count occurrences of each word



    return word_counts

