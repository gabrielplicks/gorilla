from typing import List



def str_list_to_byte_str(str_list: List[str], delimiter: str) -> str:

    """Converts a list of strings into a single string containing the byte representation of the list, separated by a delimiter.



    Args:

        str_list: A list of strings.

        delimiter: A string used to separate the byte representations of the strings in the list.

    """

    byte_str = delimiter.join(str_list).encode()

    return byte_str

