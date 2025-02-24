from typing import List, Union



def vector_sub(vector_a: List[Union[int, float]], vector_b: List[Union[int, float]]) -> Union[List[Union[int, float]], None]:

    """Subtracts vector_b from vector_a.



    Args:

        vector_a: A 2D vector.

        vector_b: A 2D vector.



    Returns:

        A list of length 2, where each element is the result of subtracting the corresponding elements from vector_a and vector_b. If the two vectors are not of equal length, returns None.

    """

    sub_vector = []

    if len(vector_a) != len(vector_b):

        return None

    else:

        for i in range(len(vector_a)):

            sub_vector.append(vector_a[i] - vector_b[i])

        return sub_vector

