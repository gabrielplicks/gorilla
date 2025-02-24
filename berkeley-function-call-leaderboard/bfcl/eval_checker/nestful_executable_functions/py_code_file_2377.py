from typing import List, Tuple, Dict



def remap_question_indices(data: List[Tuple[int, str]], remap_index: Dict[int, int]) -> List[Tuple[int, str]]:

    """Remaps the question indices in a list of tuples of question indices and answers.



    Args:

        data: A list of tuples of the question index (integer) and the corresponding answer (string).

        remap_index: A dictionary mapping the original question index to the new index.



    Returns:

        A list of tuples of the remapped question indices and answers.

    """

    remapped_data = []

    for index, answer in data:

        remapped_index = remap_index.get(index, index)

        remapped_data.append((remapped_index, answer))

    return remapped_data

