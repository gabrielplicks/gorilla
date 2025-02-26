from typing import List, Dict



def find_address_by_name(names: List[str], addresses: Dict[str, str]) -> List[str]:

    """

    Finds the address of each name in `names` using the provided `addresses` dictionary.

    Returns an empty list if `names` is empty or if any name in `names` is not present in `addresses`.



    Args:

        names: A list of names to find addresses for.

        addresses: A dictionary mapping names to addresses.

    """

    result = []



    if not names:

        return result



    for name in names:

        if name not in addresses:

            result.append(None)

        else:

            result.append(addresses[name])



    return result

