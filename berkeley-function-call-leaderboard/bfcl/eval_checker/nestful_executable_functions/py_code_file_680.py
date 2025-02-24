from typing import Dict, List



def structure_by_level(stations: Dict[str, Dict[str, List[str]]]) -> Dict[str, int]:

    """Calculates the quantity of stations by level.



    Args:

        stations: A dictionary containing station information.



    Returns:

        A dictionary with the level as a key and the number of stations as the value.

    """

    levels_count = {}

    for station_id, station_info in stations.items():

        for level in station_info['levels']:

            if level not in levels_count:

                levels_count[level] = 0

            levels_count[level] += 1

    return levels_count

