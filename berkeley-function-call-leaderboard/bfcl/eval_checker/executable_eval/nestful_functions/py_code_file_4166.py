from typing import List, Optional



def last_element(sequence: List[int]) -> Optional[int]:

    """Returns the last element of a sequence of numbers without using indexing.

    If the sequence is empty, returns None.

    Args:

        sequence: A sequence of numbers.

    """

    last = None

    for element in sequence:

        last = element

    return last

