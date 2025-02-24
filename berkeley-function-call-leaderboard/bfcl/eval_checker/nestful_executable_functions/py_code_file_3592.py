from typing import Union



def compute_similarity(str1: str, str2: str) -> float:

    """Computes the similarity between two strings, where the similarity is defined as the percentage of matching characters between the two strings.



    Args:

        str1: The first string.

        str2: The second string.



    Returns:

        The similarity as a float between 0 and 1.

    """

    if not str1 and not str2:

        return 1.0

    elif not str1 or not str2:

        return 0.0



    min_length = min(len(str1), len(str2))

    similarity = 0



    for i in range(min_length):

        if str1[i] == str2[i]:

            similarity += 1



    return similarity / min_length

