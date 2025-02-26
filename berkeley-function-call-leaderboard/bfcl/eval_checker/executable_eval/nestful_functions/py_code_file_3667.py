from typing import Dict



def filter_out_missing_values(dict_: Dict[str, object]) -> Dict[str, object]:

    """Creates a new dictionary with the same keys as the input dictionary, but with only non-null values.



    Args:

        dict_: The input dictionary.



    Returns:

        A new dictionary with the same keys as the input dictionary, but with only non-null values.

    """

    dict_filtered = {k: v for k, v in dict_.items() if v is not None}

    return dict_filtered

