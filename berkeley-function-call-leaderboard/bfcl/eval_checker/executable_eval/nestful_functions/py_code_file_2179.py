from typing import Union



def sum_of_arithmetic_series(first_term: Union[int, float], common_difference: Union[int, float], num_terms: int) -> Union[int, float]:

    """Calculates the sum of an arithmetic series with a given first term, common difference, and number of terms.



    Args:

        first_term: The first term of the series.

        common_difference: The common difference between consecutive terms.

        num_terms: The number of terms in the series.



    Returns:

        The sum of the arithmetic series.

    """

    if num_terms <= 0:

        return 0

    if common_difference == 0:

        return first_term * num_terms

    return (num_terms * (2 * first_term + (num_terms - 1) * common_difference)) // 2

