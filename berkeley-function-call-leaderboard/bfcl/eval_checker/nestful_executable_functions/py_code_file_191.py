from typing import Dict



def new_job(job_title: str, min_salary: int = 60000) -> Dict[str, str]:

    """Creates a dictionary with job title and salary.

    If the salary is less than 70000, the job title is in lowercase.

    Otherwise, the job title is in uppercase.

    Args:

        job_title: The job title.

        min_salary: The minimum salary. Defaults to 60000.

    """

    if min_salary < 70000:

        job_title = job_title.lower()

    else:

        job_title = job_title.upper()



    return {'title': job_title, 'salary': min_salary}

