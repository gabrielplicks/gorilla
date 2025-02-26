import math

from typing import List, Any



def split_batch_data(batch_data: List[List[Any]], batch_size: int) -> List[List[Any]]:

    """Splits batch data into batches of size `batch_size`.



    Args:

        batch_data: A list of lists representing the original batch data.

        batch_size: An integer indicating the desired size of each batch.



    Returns:

        A list of batches, where each batch is a list of sub-batches.

    """

    num_batches = math.ceil(len(batch_data) / batch_size)

    batches = []



    for i in range(num_batches):

        batch = []

        for j in range(batch_size):

            index = i * batch_size + j

            if index < len(batch_data):

                batch.append(batch_data[index])

        batches.append(batch)



    return batches

