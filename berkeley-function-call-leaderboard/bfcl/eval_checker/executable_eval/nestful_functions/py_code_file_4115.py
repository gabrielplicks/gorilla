from typing import List



def array_like_init(size: int) -> List[bool]:

    """Initializes an array-like object of size `size` with boolean elements.



    Boolean values are assigned based on the following rule: if the index is divisible by 3,

    the value should be True, otherwise it should be False. If the input size is less than or

    equal to 0, an empty list is returned.



    Args:

        size: The size of the array-like object to be initialized.



    Returns:

        A list of boolean values.

    """

    if size <= 0:

        return []



    return [True if i % 3 == 0 else False for i in range(size)]

