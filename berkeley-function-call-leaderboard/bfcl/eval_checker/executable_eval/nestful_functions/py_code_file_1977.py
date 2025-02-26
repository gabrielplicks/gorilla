import re

from typing import Tuple



def parse_chemical_formula(formula: str) -> Tuple[int, int]:

    """Parses a string of a chemical formula and returns a tuple of the parsed charge and multiplicity.



    Args:

        formula: The string of the chemical formula following the format of `name(m, c)`, where `name` is the name of the chemical, `m` is the multiplicity, and `c` is the charge.



    Returns:

        A tuple of the multiplicity and charge.



    Raises:

        ValueError: If the input string does not match the expected format.

    """

    match = re.search(r'(\d+), ?([-+]?\d+)', formula)



    if match:

        multiplicity = int(match.group(1))

        charge = int(match.group(2))

        return (multiplicity, charge)

    else:

        raise ValueError("Invalid chemical formula")

