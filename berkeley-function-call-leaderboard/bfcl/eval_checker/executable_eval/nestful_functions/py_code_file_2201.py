from typing import List



def count_the_word(string: str) -> List[int]:

    """Counts the occurrence of the word "the" in a given string.



    Args:

        string: The input string.



    Returns:

        A list of integers. The first integer is the index of the word "the" in the string.

        The second integer is the number of times the word "the" occurs. The third integer

        is the number of times the word "the" repeats itself in the string.

    """

    first_index = string.find("the")

    num_occurrences = string.count("the")

    num_consecutive = len(list(s for s in string.split(" ") if s == "the"))

    return [first_index, num_occurrences, num_consecutive]

