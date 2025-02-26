from typing import Dict, List



def remove_duplicate_names(input_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:

    """Removes duplicate names from the lists in the input dictionary.



    Args:

        input_dict: A dictionary with lists of names.



    Returns:

        A dictionary with the same keys as the input dictionary, but with lists that have only unique names.

    """

    output_dict = {}

    for key, value in input_dict.items():

        output_dict[key] = []

        for name in value:

            if name not in output_dict[key]:

                output_dict[key].append(name)

    return output_dict

