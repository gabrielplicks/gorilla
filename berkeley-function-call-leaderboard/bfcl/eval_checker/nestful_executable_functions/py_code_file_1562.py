from typing import List, Tuple



def extract_top_k(lst: List[Tuple[str, int]], k: int) -> List[Tuple[str, int]]:

    """Extracts the top k elements from a list of tuples, sorted in descending order by the second element.



    Args:

        lst: A list of tuples, where each tuple contains a string and an integer.

        k: The number of top elements to extract.



    Returns:

        A list of the top k elements from the input list, sorted in descending order by the second element.

    """

    top_k = sorted(lst, key=lambda x: x[1], reverse=True)[:k]

    return top_k

