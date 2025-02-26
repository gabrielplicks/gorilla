from typing import Dict, List, Any



def count_attribute(objects: List[Any], attribute: str) -> Dict[Any, int]:

    """Counts the frequency of a given attribute in a list of objects.



    Args:

        objects: A list of objects.

        attribute: The name of the attribute to count.



    Returns:

        A dictionary where the keys are the attribute values and the values are their frequencies.

    """

    frequency = {}

    for obj in objects:

        if hasattr(obj, attribute):

            value = getattr(obj, attribute)

            frequency[value] = frequency.get(value, 0) + 1

        else:

            frequency[attribute] = 1

    return frequency

