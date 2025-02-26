from typing import List



class Name:

    def __init__(self, name: str):

        self.name = name



class UpdatedName(Name):

    def __init__(self, new_name: str):

        super().__init__(new_name)



def update_all_names(names: List[Name], new_name: str) -> List[UpdatedName]:

    """Updates a list of name objects with a new name.



    Args:

        names: A list of name objects.

        new_name: The new name to update the name objects with.



    Returns:

        A list of updated name objects with the new name.

    """

    return [UpdatedName(new_name) for name in names]

