import re

from typing import Dict



def convert_camelcase_to_snakecase(d: Dict[str, str]) -> Dict[str, str]:

    """

    Converts the given dictionary's keys from camelCase to snake_case.



    Args:

        d: The dictionary to convert.



    Returns:

        A new dictionary with the keys converted to snake_case.

    """

    return {re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", k).lower(): v for k, v in d.items()}

