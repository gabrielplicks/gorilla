from typing import Dict, List



def extract_sub_dict(target_dict: Dict, keys: List[str]) -> Dict:

    """Extracts a sub-dictionary from a given dictionary.

    Args:

        target_dict: The target dictionary.

        keys: A list of keys to extract.

    Raises:

        KeyError: If any of the keys are missing from the target dictionary.

    """

    if not target_dict:

        return {}

    return {k: target_dict[k] for k in keys if k in target_dict}

