from typing import List, Tuple



def map_list_of_tuples(tuple_list: List[Tuple[str, int]]) -> List[str]:

    """Maps a list of tuples (containing a string and an integer) into a list of strings.

    Each string is constructed from the tuple's string component and an integer starting from 0.

    Args:

        tuple_list: A list of tuples, where each tuple contains a string and an integer.

    Returns:

        A list of strings, where each string is constructed from the tuple's string component and an integer starting from 0.

    """

    result = []

    for item in tuple_list:

        string_component = item[0]

        integer_component = item[1]

        concatenated_string = string_component + str(integer_component)

        result.append(concatated_string)

    return result

