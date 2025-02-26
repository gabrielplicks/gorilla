from typing import Dict, List



def common_values(dict1: Dict[str, any], dict2: Dict[str, any]) -> List[any]:

    """Finds the common values between two dictionaries.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        A list of values from the first dictionary whose keys are also present in the second dictionary.

    """

    common_values = []



    for key, value in dict1.items():

        if key in dict2:

            common_values.append(value)



    return common_values

