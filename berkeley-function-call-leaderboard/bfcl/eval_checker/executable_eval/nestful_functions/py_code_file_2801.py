from typing import List, Tuple



def check_keyword_occurrences(text: str, keyword: str) -> List[Tuple[int, int]]:

    """Checks for occurrences of a keyword in a given text and returns a list of tuples representing the starting and ending indices of each occurrence.



    Args:

        text: The input text to search for the keyword.

        keyword: The keyword to search for in the text.



    Returns:

        A list of tuples `(start_index, end_index)` representing the starting and ending indices of each occurrence of the keyword in the text.

    """

    occurrences = []

    start_index = 0

    while True:

        index = text.find(keyword, start_index)

        if index == -1:

            break

        end_index = index + len(keyword)

        occurrences.append((index, end_index))

        start_index = end_index

    return occurrences

