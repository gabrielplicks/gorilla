from typing import Dict



def count_digits_letters(string: str) -> Dict[str, int]:

    """Counts the number of digits and letters in a string.



    Args:

        string: The input string.



    Returns:

        A dictionary with the counts for digits and letters.

    """

    counts = {

        'digits': 0,

        'letters': 0,

    }



    for c in string:

        if c.isdigit():

            counts['digits'] += 1

        elif c.isalpha():

            counts['letters'] += 1



    return counts

