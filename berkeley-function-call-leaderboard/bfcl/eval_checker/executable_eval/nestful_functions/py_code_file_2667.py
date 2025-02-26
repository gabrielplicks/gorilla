from typing import AnyStr



def countSegments(s: AnyStr) -> int:

    """Counts the number of segments in a string.



    A segment is defined to be a contiguous sequence of non-space characters.



    Args:

        s: The string to be segmented.



    Returns:

        The number of segments in the string.

    """

    return len(s.split())

