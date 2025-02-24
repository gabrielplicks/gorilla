from typing import Dict, List



def create_output_dictionary(data: Dict[str, List[int]]) -> Dict[str, List[int]]:

    """Constructs a dictionary with the desired format from a given dictionary.



    Args:

        data: A dictionary mapping each pair of names to a list of numbers.



    Returns:

        out: A dictionary with the desired format.

    """

    out = {'Names': sorted(data.keys())}

    for name in out['Names']:

        out[name] = data[name]

    return out

