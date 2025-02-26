import random

random.seed(42)
import typing



def draw_names(names: typing.List[str], k: int) -> typing.List[str]:

    """Selects `k` unique names from a list of names.



    Args:

        names: A list of names.

        k: The number of names to select.



    Raises:

        ValueError: If `k` is greater than the length of `names`.

    """

    if k > len(names):

        raise ValueError("Cannot select more names than the list contains.")



    random_names = random.sample(names, k)

    return random_names

