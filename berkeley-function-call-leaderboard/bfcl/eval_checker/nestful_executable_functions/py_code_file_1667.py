from typing import List, Dict



def compute_gpu_count(tasks: List[str], gpu_counts: Dict[str, int]) -> int:

    """Computes the total number of GPUs required by a job.



    Args:

        tasks: A list of tasks.

        gpu_counts: A dictionary of GPU counts per task.



    Returns:

        The total number of GPUs required by the job.

    """

    gpu_total = 0



    for task in tasks:

        gpu_count = gpu_counts.get(task, 1)

        gpu_total += gpu_count



    return gpu_total

