import itertools

from typing import Dict, List, Union



def combinations_of_lists(input_dict: Dict[str, List[Union[int, str]]]) -> List[Dict[str, Union[int, str]]]:

    """

    Generates all possible combinations of elements from each list in the input dictionary.



    Args:

        input_dict: A dictionary where the keys are strings and the values are lists of integers or strings.



    Returns:

        A list of dictionaries where each dictionary contains a combination of elements from the input dictionary.

    """

    return [{key: value for key, value in zip(input_dict.keys(), values)}

            for values in itertools.product(*input_dict.values())]

