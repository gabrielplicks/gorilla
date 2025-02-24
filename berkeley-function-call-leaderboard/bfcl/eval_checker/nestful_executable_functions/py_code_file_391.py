from typing import Dict, Union



def get_rig_type(rig_abbreviation: str) -> str:

    """Returns the rig type based on the given rig abbreviation.



    Args:

        rig_abbreviation: The abbreviation of the rig type.



    Raises:

        ValueError: If the rig abbreviation is not one of "B", "C", "DR", "H", or "T".

    """

    rig_types: Dict[str, str] = {

        "B": "bender",

        "C": "crusher",

        "DR": "drilling rig",

        "H": "hoist",

        "T": "tanker"

    }



    if rig_abbreviation not in rig_types:

        raise ValueError("Invalid rig abbreviation.")



    return rig_types[rig_abbreviation]

