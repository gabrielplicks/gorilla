from typing import List



def get_precise_rating_coefficient(ratings: List[int]) -> float:

    """Calculates the average of the top 75% of the ratings.



    Args:

        ratings: A list of ratings.



    Returns:

        The average of the top 75% of the ratings as a floating point number.

    """

    sorted_ratings = sorted(ratings)

    num_top_ratings = int(len(sorted_ratings) * 0.75)

    top_ratings = sorted_ratings[-num_top_ratings:]

    average = sum(top_ratings) / len(top_ratings)

    return average

