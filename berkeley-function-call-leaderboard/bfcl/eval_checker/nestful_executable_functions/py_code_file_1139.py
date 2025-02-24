from typing import List



class Car:

    def __init__(self, make: str, model: str, year: int):

        self.make = make

        self.model = model

        self.year = year



def cars_by_make(cars: List[Car], make: str) -> List[Car]:

    """Returns a list of Car objects with the given make.



    Args:

        cars: A list of Car objects.

        make: A string representing the make of the cars to return.

    """

    return [car for car in cars if car.make == make]

