from typing import Union



def classify_road(material_code: Union[int, None], road_type: Union[str, None]) -> str:

    """Classifies the road type into one of three categories: "National", "State", or "Other".



    Args:

        material_code: The material code of the road, an integer between 1 and 6.

        road_type: The type of the road.



    Returns:

        The category of the road.

    """

    if not 1 <= material_code <= 6:

        return "Unknown"

    if road_type is None:

        return "Unknown"



    if "National" in road_type and material_code in (1, 2, 3):

        return "National"

    elif "State" in road_type and material_code in (1, 2, 3):

        return "State"

    else:

        return "Other"

