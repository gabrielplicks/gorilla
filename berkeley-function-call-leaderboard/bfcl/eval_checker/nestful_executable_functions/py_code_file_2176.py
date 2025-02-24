from typing import List, Tuple



def create_tasks(input_list: List[Tuple], num_tasks: int) -> List[List[Tuple]]:

    """Creates a list of tasks from a given list of tuples.



    Each task is a list of tuples, created by slicing the input list.

    The number of tuples in each task is determined by the `num_tasks` parameter.



    Args:

        input_list: A list of tuples.

        num_tasks: The number of tuples in each task.



    Returns:

        A list of tasks.

    """

    tasks = []

    for i in range(0, len(input_list), num_tasks):

        tasks.append(input_list[i:i+num_tasks])

    return tasks

