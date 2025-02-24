from typing import List



def find_indices_of_words(words: List[str]) -> List[List[int]]:

    """Finds the indices of words in a list of strings.



    Args:

        words: A list of strings.



    Returns:

        A list of lists of integers, where each inner list contains the indices of the words in the corresponding string of the input list.

    """

    output = []

    for word in words:

        try:

            indices = [index for index, _ in enumerate(word)]

            output.append(indices)

        except ValueError:

            pass

    return output

