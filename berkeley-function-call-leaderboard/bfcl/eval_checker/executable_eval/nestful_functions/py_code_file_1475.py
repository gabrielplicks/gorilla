from typing import List



def calculate_impact_factor(citations: List[int]) -> int:

    """Calculates the citation impact factor of a scientific paper.



    The function takes a list of citation counts of a paper and returns the calculated impact factor.

    The impact factor is calculated by dividing the sum of all citation counts by the number of citation

    counts, rounded to the nearest whole number. If the citation count list is empty, return 0.



    Args:

        citations: A list of citation counts of a paper.



    Returns:

        The calculated impact factor.

    """

    if not citations:

        return 0

    impact_factor = sum(citations) // len(citations)

    return round(impact_factor)

