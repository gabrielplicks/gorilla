from typing import Dict



def get_fetal_weight(input_dict: Dict[str, str]) -> float:

    """Calculates the fetal weight based on the input dictionary.



    Args:

        input_dict: A dictionary containing the gestational age, birth weight, and whether the patient had a Caesarian section.



    Returns:

        The fetal weight as a float.

    """

    gestational_age = float(input_dict["gestational_age_at_birth"].replace(" weeks", ""))

    birth_weight = float(input_dict["birth_weight"].replace(" grams", ""))

    is_ceasarian = input_dict["is_ceasarian"]



    if is_ceasarian:

        return 0.75 * birth_weight * gestational_age

    else:

        return birth_weight * gestational_age

