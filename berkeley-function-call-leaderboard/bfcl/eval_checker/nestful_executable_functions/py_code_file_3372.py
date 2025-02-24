from typing import List



class Dummy:

    def __init__(self, name: str, age: int):

        self.name = name

        self.age = age



def create_list_of_dummy_objects(names: List[str]) -> List[Dummy]:

    """Constructs a list of dummy objects from a list of names.



    Each object has `.name` and `.age` attributes. The `.age` attribute is set to `0` for all objects.



    Args:

        names: A list of names to use for the objects.



    Returns:

        A list of dummy objects.

    """

    return [Dummy(name, 0) for name in names]

