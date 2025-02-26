from typing import Dict, List



def filter_words(dictionary: Dict[str, List[str]], freq_array: List[List[int]]) -> Dict[str, List[str]]:

    """Filters the words in the input dictionary whose frequency of occurrence is greater than or equal to 100.



    Args:

        dictionary: A dictionary with key-value pairs where the keys are unicode strings and the values are lists of unicode strings.

        freq_array: A 2D array that represents the frequency of occurrence of each word in the dictionary.



    Returns:

        A dictionary with the filtered words from the input dictionary whose frequency of occurrence is greater than or equal to 100.

    """

    filtered_dictionary = {word: freq_list for word, freq_list in dictionary.items() if freq_list[0] >= 100}

    return filtered_dictionary

