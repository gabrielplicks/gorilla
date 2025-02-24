import re

from typing import List, Tuple



def extract_currency_from_string(text: str) -> List[Tuple[str, str]]:

    """Extracts currency values and their corresponding names from a string.



    Args:

        text: The input string.



    Returns:

        A list of tuples containing the extracted currency values and their corresponding names.

    """

    currency_regex = r"\$(\d+\.?\d*) ?(\w+)"

    matches = re.findall(currency_regex, text)



    currency_values = []

    for match in matches:

        amount, name = match

        currency_values.append((f"${amount} {name}", amount))



    return currency_values

