from typing import List



def sort_cities_by_population(cities: List[str], populations: List[float]) -> (List[str], List[float]):

    """Sorts two lists of cities and their corresponding populations based on the population of each city in descending order.



    Args:

        cities: A list of city names.

        populations: A list of corresponding city populations.



    Returns:

        A tuple containing the sorted lists of cities and populations.

    """

    sorted_cities = sorted(cities, key=lambda city: populations[cities.index(city)], reverse=True)

    sorted_populations = sorted(populations, reverse=True)

    return sorted_cities, sorted_populations

