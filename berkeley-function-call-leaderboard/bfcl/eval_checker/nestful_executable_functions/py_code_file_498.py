from typing import Dict



def update_ADT(adt: Dict[str, int]) -> None:

    """Updates all of the attributes of an ADT (abstract data type) by adding one to each numeric attribute.



    Args:

        adt: The ADT to update.

    """

    for key, value in adt.items():

        if isinstance(value, int):

            adt[key] += 1

        elif isinstance(value, dict):

            update_ADT(value)

