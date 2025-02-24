import re

from typing import List



def get_jira_test_case_keys(string: str) -> List[str]:

    """Extracts Jira test case keys from a given string using a regex pattern.



    Args:

        string: The input string to search for Jira test case keys.



    Returns:

        A list of Jira test case keys found in the input string.

    """

    pattern = r"PROJECT-\d{4}"  # Regex pattern to match Jira test case keys

    matches = re.findall(pattern, string)

    return matches

