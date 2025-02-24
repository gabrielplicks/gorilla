import asyncio

from typing import List



async def task(task_id: int) -> str:

    """Simulates an asynchronous task and returns a completion message.



    Args:

        task_id: The ID of the task.



    Returns:

        A completion message for the task.

    """

    await asyncio.sleep(1)  # Simulate an asynchronous task

    return f"Task {task_id} completed"



async def run_tasks(task_count: int) -> List[str]:

    """Runs a set of asynchronous tasks concurrently and returns a list of results.



    Args:

        task_count: The number of tasks to run.



    Returns:

        A list of results from the tasks.

    """

    tasks = [task(i) for i in range(task_count)]

    results = await asyncio.gather(*tasks)

    return results

