from typing import List



def reverse_vowels(string: str) -> str:

    """Reverses the vowels in a given string.



    The vowels to be reversed are 'a', 'e', 'i', 'o', and 'u'. The function also preserves the case of the vowels.



    Args:

        string: The input string.



    Returns:

        The string with the vowels reversed.

    """

    vowels: List[str] = ['a', 'e', 'i', 'o', 'u']

    stack: List[str] = []



    for char in string:

        if char.lower() in vowels:

            stack.append(char)

    reverse_string: str = ''

    for char in string:

        if char.lower() in vowels:

            reverse_string += stack.pop()

        else:

            reverse_string += char



    return reverse_string

