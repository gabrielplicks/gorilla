from typing import Any



def is_positive_or_negative(obj: Any) -> bool:

    """Checks if the `.label` attribute of the input object is either `'positive'` or `'negative'`.



    Args:

        obj: The input object.



    Returns:

        True if the `.label` attribute of the input object is either `'positive'` or `'negative'`, False otherwise.

    """

    return obj.label in ('positive', 'negative')

