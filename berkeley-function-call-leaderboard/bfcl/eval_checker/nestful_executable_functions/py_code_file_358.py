from typing import Dict, Union



def compare_a_and_b(dictionary: Dict[str, Union[int, None]]) -> Union[bool, None]:

    """

    Returns `True` if the `b` value is greater than the `a` value, `False` if the `b` value is less than the `a` value,

    and `None` if either `a` or `b` is missing.

    Args:

        dictionary: A dictionary containing two fields, `a` and `b`.

    """

    if 'a' not in dictionary or 'b' not in dictionary:

        return None



    if dictionary['b'] > dictionary['a']:

        return True

    elif dictionary['b'] < dictionary['a']:

        return False

    else:

        return None

