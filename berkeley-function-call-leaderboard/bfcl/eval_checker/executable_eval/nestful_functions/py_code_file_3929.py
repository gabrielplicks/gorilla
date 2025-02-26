from typing import Union



def calculate_biomass_offtake_rate(biomass_availability: Union[int, float], biomass_offtake_rate: Union[int, float], management_threshold: Union[int, float]) -> Union[int, float]:

    """Calculates the biomass offtake rate based on the availability of biomass and the management threshold.



    Args:

        biomass_availability: The biomass availability.

        biomass_offtake_rate: The biomass offtake rate.

        management_threshold: The management threshold.



    Returns:

        The biomass offtake rate, which is equal to the biomass availability if it is above the management threshold, but is zero otherwise.

    """

    if biomass_availability > management_threshold:

        return biomass_availability

    else:

        return 0

