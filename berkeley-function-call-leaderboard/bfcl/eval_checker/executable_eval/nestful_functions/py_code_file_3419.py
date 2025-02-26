from typing import List, Tuple



def get_priority_tasks(tasks: List[Tuple[int, int]], priority: int) -> List[Tuple[int, int]]:

    """Returns a list of tuples of the form (task_id, task_value) where task_id is the index of the task in the original list, task_value is the value of the task, and the priority of the task is at least priority.



    Args:

        tasks: A list of tuples of the form (value, priority) that represent tasks with different values and priorities.

        priority: The minimum priority to filter for.



    Returns:

        A list of tuples of the form (task_id, task_value).

    """

    return [(i, tasks[i][0]) for i in range(len(tasks)) if tasks[i][1] >= priority]

