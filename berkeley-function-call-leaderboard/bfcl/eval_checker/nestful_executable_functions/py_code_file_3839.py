from typing import List



def get_anagram_groups(letters: str) -> List[str]:

    """

    Returns a list of strings that represent all possible anagram groups.

    Two strings belong to the same anagram group if they can be constructed

    from the same set of letters.



    Args:

        letters: A string of lowercase letters.



    Returns:

        A list of strings that represent all possible anagram groups.

    """

    if len(letters) == 0:

        return [""]

    anagram_groups = {}

    for i in range(len(letters)):

        letter = letters[i]

        remaining_letters = letters[:i] + letters[i+1:]

        for group in get_anagram_groups(remaining_letters):

            anagram_groups.setdefault(group+letter, []).append(letter)

    return list(anagram_groups.keys())

