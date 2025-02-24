from typing import Dict



def update_attributes(attributes: Dict[str, bool]) -> None:

    """

    Updates the attributes dictionary based on the given conditions.



    Args:

        attributes: A dictionary containing the "found" and "verified" attributes.



    Raises:

        ValueError: If the "found" attribute is True and the "verified" attribute is False.

    """

    if attributes.get("found") and (attributes.get("verified") == False or not attributes.get("verified")):

        attributes["verified"] = True

    elif attributes.get("verified") == False and (attributes.get("found") or not attributes.get("found")):

        attributes["found"] = False

    elif attributes.get("found") and attributes.get("verified") == False:

        raise ValueError("Attributes are not valid")

