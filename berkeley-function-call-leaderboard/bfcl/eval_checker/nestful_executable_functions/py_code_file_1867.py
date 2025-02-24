from typing import Tuple



def tuple_to_dict(tuple_of_strings: Tuple[str, str, str]) -> dict:

    """Creates a dictionary from a tuple of strings.



    The keys of the dictionary are the first two elements of the tuple and the values are the third element.



    Args:

        tuple_of_strings: A tuple of strings.



    Returns:

        A dictionary.

    """

    dictionary = {}



    if len(tuple_of_strings) < 3:

        return dictionary



    first_key = tuple_of_strings[0]

    second_key = tuple_of_strings[1]

    value = tuple_of_strings[2]



    dictionary[first_key] = second_key

    dictionary[second_key] = value



    return dictionary

