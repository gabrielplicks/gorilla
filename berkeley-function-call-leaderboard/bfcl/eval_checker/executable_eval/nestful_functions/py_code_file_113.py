import re

from typing import Dict



def extract_projection_parameters(wkt_string: str) -> Dict[str, float]:

    """Extracts the spatial projection parameters from a WKT string representing a geometry.

    If the WKT string does not contain the projection parameters, the function returns None.

    Args:

        wkt_string: The WKT string representing a geometry.

    Returns:

        A dictionary containing the projection parameters, or None if the WKT string does not contain the projection parameters.

    """

    projection_parameters_regex = r'PARAMETER\["(.*?)",(.*?)\]'

    projection_parameters_matches = re.findall(projection_parameters_regex, wkt_string)

    if len(projection_parameters_matches) == 0:

        return None

    projection_parameters = {}

    for match in projection_parameters_matches:

        projection_parameters[match[0]] = float(match[1])

    return projection_parameters

