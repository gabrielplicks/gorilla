from typing import List



def find_cat_dog(strings: List[str]) -> List[str]:

    """Finds all strings in a list that contain the substring "cat" or "dog".



    Args:

        strings: A list of strings to search.



    Returns:

        A new list containing all strings that contain either "cat" or "dog" in the `strings` list.

    """

    return [s for s in strings if "cat" in s] + [s for s in strings if "dog" in s]

