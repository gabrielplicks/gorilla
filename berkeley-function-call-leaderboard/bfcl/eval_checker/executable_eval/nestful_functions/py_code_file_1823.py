from typing import List



def remove_duplicates_using_dictionary(string_list: List[str]) -> List[str]:

    """Removes duplicate strings from a list using a dictionary.



    Args:

        string_list: A list of strings.



    Returns:

        A new list with duplicate strings removed.

    """

    frequencies = {}

    for string in string_list:

        if string not in frequencies:

            frequencies[string] = 1

        else:

            frequencies[string] += 1

    return [string for string, frequency in frequencies.items() if frequency == 1]

