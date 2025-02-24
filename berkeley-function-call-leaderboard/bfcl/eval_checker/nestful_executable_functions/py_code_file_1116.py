from typing import Tuple, Optional



def pack_split_params(s: str) -> Tuple[str, str, Optional[str]]:

    """Packs the substrings of a comma-separated string into a tuple.



    Args:

        s: The input string.



    Returns:

        A tuple containing the substrings of the input string.

    """

    pieces = s.split(',')

    if len(pieces) > 2:

        named_params = pieces[-1]

        pieces = pieces[:-1]

    else:

        named_params = None

    arg1, arg2 = pieces

    return tuple([arg1, arg2, named_params])

