from typing import Set



def is_heavy_atom(chemical_formula: str) -> bool:

    """Checks whether a given chemical formula contains any heavy atom.



    Args:

        chemical_formula: A string representing a chemical formula.



    Returns:

        A boolean value `True` if the formula contains at least one heavy atom,

        and `False` otherwise.

    """

    heavy_atoms: Set[str] = {'C', 'N', 'O', 'S', 'P'}

    formula_atoms = chemical_formula.replace('(', ' ').replace(')', ' ').split()

    for atom in formula_atoms:

        if atom[0] in heavy_atoms:

            return True

    return False

