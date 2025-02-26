from typing import List, Dict



class Car:

    def __init__(self, make: str, model: str):

        self.make = make

        self.model = model



def get_cars_by_make(cars: List[Car]) -> Dict[str, List[Car]]:

    """

    Returns a dictionary of Cars keyed by their make.



    Args:

        cars: A list of Cars.

    """

    cars_by_make = {}

    for car in cars:

        make = car.make

        if make not in cars_by_make:

            cars_by_make[make] = []

        cars_by_make[make].append(car)

    return cars_by_make

