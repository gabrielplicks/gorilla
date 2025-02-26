from typing import List, Tuple, Dict



def make_dictionary_from_pairs_and_values(pairs: List[Tuple[str, int]], dictionary: Dict[str, int]) -> Dict[str, int]:

    """Creates a dictionary from a list of key-value pairs and a dictionary.



    If a key from the pair already exists in the dictionary, it prints a warning message.



    Args:

        pairs: A list of key-value pairs.

        dictionary: A dictionary to be updated.



    Returns:

        The updated dictionary.

    """

    for key, value in pairs:

        if key in dictionary:

            print(f"Warning: Key {key} already exists in the dictionary")

        else:

            dictionary[key] = value



    return dictionary

