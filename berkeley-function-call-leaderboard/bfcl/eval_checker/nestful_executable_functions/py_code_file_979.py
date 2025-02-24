from typing import Dict, List



def get_station_data(filename: str) -> Dict[str, List[float]]:

    """Reads a CSV file and returns a dictionary of lists containing station data.



    Args:

        filename: The name of the CSV file to read.



    Returns:

        A dictionary with keys 'id', 'name', 'lat', and 'lon' and values as lists of corresponding values for each station.

    """

    with open(filename, 'r') as f:

        lines = f.readlines()



    data = {

        'id': [],

        'name': [],

        'lat': [],

        'lon': [],

    }



    for line in lines[1:]:

        id, name, lat, lon = line.strip().split(',')

        data['id'].append(int(id))

        data['name'].append(name)

        data['lat'].append(float(lat))

        data['lon'].append(float(lon))



    return data

