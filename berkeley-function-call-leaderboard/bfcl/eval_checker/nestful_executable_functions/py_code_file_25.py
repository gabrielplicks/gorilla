from typing import List, Dict



def get_job_status(queue: List[Dict], job_name: str) -> str:

    """

    Returns the status of a job in the given job queue.



    Args:

        queue: A list of dictionaries representing the job queue. Each dictionary should have 'name' and 'status' keys.

        job_name: The name of the job to find.



    Returns:

        The status of the job if found, otherwise "Job not found."

    """

    for job in queue:

        if job['name'] == job_name:

            return job['status']

    return "Job not found."

