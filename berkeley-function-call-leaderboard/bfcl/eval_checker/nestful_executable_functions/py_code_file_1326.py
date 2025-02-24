from typing import List



def sort_strings_by_number(input_list: List[str]) -> List[str]:

    """Sorts a list of strings by their corresponding numbers.



    The numbers are separated from the preceding string by a space, are positive integers with no leading zeros,

    and are sorted in descending order. Ties are broken by lexicographical order of the strings.



    Args:

        input_list: The list of strings to be sorted.



    Returns:

        The sorted list of strings.

    """

    def key_func(string: str) -> int:

        """Extracts the number from the string.



        Args:

            string: The string to extract the number from.



        Returns:

            The number as an integer.

        """

        return int(string.split()[-1])



    return sorted(input_list, key=key_func, reverse=True)

