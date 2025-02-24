from typing import Dict



def convert_dict_to_paths(input_dict: Dict[str, Dict[str, int]]) -> List[str]:

    """Converts a dictionary of dictionaries to a list of file paths.



    Args:

        input_dict: A dictionary where the keys are strings and the values are dictionaries.

            The inner dictionaries have string keys and integer values.



    Returns:

        A list of file paths as strings.

    """

    file_paths = []



    for outer_key, inner_dict in input_dict.items():

        for inner_key, value in inner_dict.items():

            file_paths.append(f'{outer_key}/{inner_key}')



    return file_paths

