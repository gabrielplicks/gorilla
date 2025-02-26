from typing import List



def sort_numbers_in_list(s: str) -> List[int]:

    """Sorts a list of integers from a string.



    Args:

        s: A string containing a list of integers separated by commas.



    Returns:

        A sorted list of integers.



    Raises:

        ValueError: If the string does not contain integers only.

    """

    l = s.split(",")

    for i in range(len(l)):

        try:

            l[i] = int(l[i])

        except ValueError:

            raise ValueError("The string does not contain integers only.")

    return sorted(l)

