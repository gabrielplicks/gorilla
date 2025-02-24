import random

random.seed(42)
from typing import List, Optional



def random_item(lst: List[Optional[int]]) -> Optional[int]:

    """Chooses a random item from a list. If the input list is empty, returns None.



    Args:

        lst: The input list.

    """

    if lst:

        return random.choice(lst)

    else:

        return None

