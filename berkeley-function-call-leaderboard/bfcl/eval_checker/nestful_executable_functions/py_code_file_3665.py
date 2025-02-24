from typing import Dict, List



def mask_dictionary_values(dictionary: Dict[str, str], keys: List[str]) -> Dict[str, str]:

    """Masks the values of a dictionary for a given list of keys.



    Args:

        dictionary: The dictionary to mask.

        keys: The keys to mask.

    """

    masked_dict = copy.deepcopy(dictionary)

    for key in keys:

        masked_dict[key] = "****"

    return masked_dict

