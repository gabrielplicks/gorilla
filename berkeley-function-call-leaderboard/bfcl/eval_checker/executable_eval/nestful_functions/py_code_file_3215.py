from typing import List



class Object:

    def __init__(self, name: str):

        self.name = name



def count_objects_with_name_starting_with_a_or_b(objects: List[Object]) -> int:

    """Counts the number of objects whose `name` attribute starts with 'A' or 'B'.



    Args:

        objects: A list of objects.



    Returns:

        The number of objects whose `name` attribute starts with 'A' or 'B'.

    """

    count = 0

    for obj in objects:

        if obj.name.startswith('A') or obj.name.startswith('B'):

            count += 1

    return count

