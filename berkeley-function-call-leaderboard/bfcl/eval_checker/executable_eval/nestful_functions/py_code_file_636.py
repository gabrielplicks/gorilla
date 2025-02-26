from typing import Dict, List



def digit_count(string_list: List[str]) -> Dict[str, int]:

    """Computes the count of the number of occurrences of each digit in a list of strings.



    Args:

        string_list: A list of strings containing only digits.



    Returns:

        A dictionary in which each key is a digit and the corresponding value is the count of occurrences of that digit.

    """

    digit_counts: Dict[str, int] = {}

    for string in string_list:

        if string.isdigit():

            digit_counts[string] = digit_counts.get(string, 0) + 1

    return digit_counts

