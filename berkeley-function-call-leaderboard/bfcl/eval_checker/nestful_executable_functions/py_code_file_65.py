from typing import Dict, Tuple



def get_top_n_pairs_from_dict(d: Dict[str, int], n: int) -> list[Tuple[str, int]]:

    """Returns the top n key-value pairs from a dictionary in descending order of values.



    Args:

        d: The dictionary to extract the pairs from.

        n: The number of top pairs to return.

    """

    sorted_dict = sorted(d.items(), key=lambda item: item[1], reverse=True)

    pairs = [(key, value) for key, value in sorted_dict]

    return pairs[:n]

