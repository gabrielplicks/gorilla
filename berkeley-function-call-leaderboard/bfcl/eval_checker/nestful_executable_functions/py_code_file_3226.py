from typing import Dict, List



def get_adult_names(age_dict: Dict[str, int]) -> List[str]:

    """Returns a list of names whose age is greater than or equal to 18.



    Args:

        age_dict: A dictionary of names to ages.

    """

    return [name for name, age in age_dict.items() if age >= 18]

