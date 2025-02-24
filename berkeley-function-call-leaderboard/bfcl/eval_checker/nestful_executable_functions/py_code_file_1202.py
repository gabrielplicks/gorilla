from typing import List



def calculate_vector_sum(vectors: List[List[float]]) -> List[float]:

    """Calculates the sum of all vectors in a list as a single vector.



    Args:

        vectors: A list of vectors, where each vector is a list of numbers.



    Returns:

        The sum of all vectors in the list as a single vector.

    """

    sum_vector = [0] * len(vectors[0])  # Initialize the sum vector with zeroes of the same length as the first vector in the list



    for vector in vectors:

        for idx, element in enumerate(vector):

            sum_vector[idx] += element



    return sum_vector

