from typing import List



def k_mers(string: str, k: int) -> List[str]:

    """Generates all k-mers of a given input string.



    Args:

        string: The input string.

        k: The length of the k-mers.



    Returns:

        A list of k-mers.

    """

    k_mers = []

    for i in range(len(string) - k + 1):

        k_mers.append(string[i:i + k])



    return k_mers

