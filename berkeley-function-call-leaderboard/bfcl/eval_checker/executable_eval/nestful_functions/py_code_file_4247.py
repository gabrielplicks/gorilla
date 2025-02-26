from typing import List, Tuple



def filter_duplicates(tuples: List[Tuple[int, str]]) -> List[Tuple[int, str]]:

    """Filters out the tuples with duplicate first elements.



    Args:

        tuples: A list of tuples.



    Returns:

        A new list that contains only the tuples with unique first element.

    """

    result = []

    seen_first_elements = set()



    for tup in tuples:

        if tup[0] in seen_first_elements:

            continue

        seen_first_elements.add(tup[0])

        result.append(tup)



    return result

