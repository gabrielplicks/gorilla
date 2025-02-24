from typing import List, Tuple



def generate_symbol_pairs(symbols: List[str]) -> List[Tuple[str, str]]:

    """Generates all possible pairs of symbols from a given list of symbols, including duplicates.



    Args:

        symbols: A list of symbols.



    Returns:

        A list of pairs of symbols as tuples.

    """

    if not symbols:

        return []

    pairs = [(s1, s2) for s1 in symbols for s2 in symbols]



    return pairs

