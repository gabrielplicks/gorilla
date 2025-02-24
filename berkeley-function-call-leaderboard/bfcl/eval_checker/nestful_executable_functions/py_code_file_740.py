from collections import defaultdict

from typing import Dict, List



def string_indices_dict(input_list: List[str]) -> Dict[str, List[int]]:

    """Creates a dictionary where the keys are the unique strings in the input list and the values are a list of indices where each string appears.



    Args:

        input_list: A list of strings.



    Returns:

        A dictionary where the keys are the unique strings in the input list and the values are a list of indices where each string appears.

    """

    output_dict = defaultdict(list)

    for i, string in enumerate(input_list):

        output_dict[string].append(i)

    return output_dict

