from typing import List, Optional



def integer_list_to_string(integer_list: List[int], file_path: Optional[str] = None) -> Optional[str]:

    """Converts a list of integers into a string representing a space-separated sequence of those integers.

    If a file path is provided as an optional parameter, the function saves the string in that file.

    If the file path is not provided, the function returns the string.



    Args:

        integer_list: A list of integers to be converted into a string.

        file_path: An optional file path to save the string.



    Returns:

        The string representing the space-separated sequence of integers, or None if the file path is provided.

    """

    string = " ".join([str(integer) for integer in integer_list])

    if file_path:

        with open(file_path, "w") as file:

            file.write(string)

    else:

        return string

