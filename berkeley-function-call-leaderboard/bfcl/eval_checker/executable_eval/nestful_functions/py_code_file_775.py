from typing import List, Set



def uncommon_words(s1: str, s2: str) -> List[str]:

    """Returns a sorted list of words that are unique to either string.



    Args:

        s1: The first string.

        s2: The second string.



    Returns:

        A sorted list of words that are unique to either string.

    """

    return sorted({word for word in set(s1.split()) | set(s2.split()) if word not in set(s1.split()) & set(s2.split())})

