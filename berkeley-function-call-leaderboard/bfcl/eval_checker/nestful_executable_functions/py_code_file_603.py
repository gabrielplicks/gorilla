from typing import List, Any



def validate_task_objects(objects: List[Any]) -> bool:

    """Validates task objects to ensure they have valid values for their task_id, name, and description attributes.



    Args:

        objects: A list of task objects to validate.



    Returns:

        True if all objects in the list have valid values for their task_id, name, and description attributes. False otherwise.

    """

    for task in objects:

        if not isinstance(task.task_id, str) or len(task.task_id) < 1 or len(task.task_id) > 128:

            return False

        if not isinstance(task.name, str) or len(task.name) < 1 or len(task.name) > 100:

            return False

        if not isinstance(task.description, str) or len(task.description) > 100:

            return False



    return True

