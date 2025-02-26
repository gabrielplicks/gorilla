from collections import defaultdict

from typing import List



def get_anagrams(strings: List[str]) -> List[List[str]]:

    """

    Returns a list of lists of strings that are anagrams of each other.



    Args:

        strings: A list of strings.

    """

    anagram_groups = defaultdict(list)

    for string in strings:

        sorted_string = "".join(sorted(string))

        anagram_groups[sorted_string].append(string)



    return [group for group in anagram_groups.values() if len(group) > 1]

