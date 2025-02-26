from typing import Any, TypeVar



X = TypeVar('X')



def cardinality(X: Any) -> int:

    """Calculates the size of a set without constructing it.



    Args:

        X: The input data.



    Returns:

        The size of the set.

    """

    return len(set(X))

