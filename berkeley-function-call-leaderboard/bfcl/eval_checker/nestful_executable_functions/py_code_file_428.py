import json

from typing import Dict, Any



def update_car_prices(file_path: str) -> None:

    """Updates the car prices in a JSON file to 0.0.



    Args:

        file_path: The path to the JSON file.

    """

    with open(file_path, 'r') as file:

        cars: Dict[str, Any] = json.load(file)



    for car in cars:

        car['carPrice'] = 0.0



    with open(file_path, 'w') as file:

        json.dump(cars, file)

