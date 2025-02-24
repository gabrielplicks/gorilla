from typing import Dict, List, Tuple



def group_tuples_by_age(tuples: List[Tuple[str, int]]) -> Dict[int, List[str]]:

    """Groups a list of tuples containing a name and an age by age and sorts them by name.



    Args:

        tuples: A list of tuples containing a name and an age.



    Returns:

        A dictionary where the keys are the ages and the values are lists of names sorted by name.

    """

    age_dict = {}

    for name, age in tuples:

        if age not in age_dict:

            age_dict[age] = []

        age_dict[age].append(name)



    for age in age_dict:

        age_dict[age] = sorted(age_dict[age])



    return age_dict

