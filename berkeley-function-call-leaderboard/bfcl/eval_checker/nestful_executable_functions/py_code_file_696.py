from typing import Dict, Optional



def get_population(country: str, populations: Dict[str, int]) -> Optional[int]:

    """Retrieve the population of a country from a dictionary.



    Args:

        country: A string representing the country name.

        populations: A dictionary containing country names as keys and population data as values.



    Returns:

        The population data for the specified country, or None if the country is not in the dictionary.



    Example:

        >>> populations = {'China': 1439323776, 'India': 1380004385, 'United States': 331002651}

        >>> get_population('China', populations)

        1439323776

        >>> get_population('United Kingdom', populations) is None

        True

    """

    if country in populations:

        return populations[country]

    else:

        return None

