from typing import List



def create_square_tensor(n: int) -> List[List[int]]:

    """Creates a square tensor of size `n` whose diagonal entries are 1 and all other entries are 0.



    Args:

        n: The size of the tensor.

    """

    tensor = [[0] * n for _ in range(n)]

    for i in range(n):

        for j in range(n):

            if i == j:

                tensor[i][j] = 1

    return tensor

