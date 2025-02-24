import datetime

import subprocess



def generate_commit_message() -> str:

    """Generates a commit message based on the current branch name and date.



    The function retrieves the current branch name using the `git rev-parse --abbrev-ref HEAD` command and

    constructs the commit message by concatenating the branch name and the current date in the format

    `branch-name: YYYY-MM-DD`.



    Returns:

        The generated commit message.

    """

    current_branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode('utf-8').strip()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    commit_message = f'{current_branch}: {current_date}'

    return commit_message

