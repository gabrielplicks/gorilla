from typing import List



def count_letters_and_digits(string: str) -> tuple:

    """Counts the number of letters and digits in a given string.



    Args:

        string: The input string.



    Returns:

        A tuple containing the number of letters and digits in the string.

    """

    letters: List[str] = [char for char in string if char.isalpha()]

    digits: List[str] = [char for char in string if char.isdigit()]

    return len(letters), len(digits)

