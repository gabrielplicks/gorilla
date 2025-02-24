from typing import List



def is_all_hydrogen(atomic_numbers: List[int]) -> bool:

    """Determines if all atomic numbers of an atom type in a molecule are hydrogen atoms.



    Args:

        atomic_numbers: A list of atomic numbers.



    Returns:

        True if all atomic numbers are hydrogen atoms, False otherwise.

    """

    for atomic_number in atomic_numbers:

        if atomic_number != 1:

            return False

    return True

