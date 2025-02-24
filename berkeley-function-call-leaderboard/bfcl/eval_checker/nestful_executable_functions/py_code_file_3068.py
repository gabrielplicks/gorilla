from typing import List



def clean_emails(email_string: str) -> List[str]:

    """

    Takes a string that represents a comma-separated list of valid emails and returns a list of those emails.

    The function cleans each email by removing surrounding whitespaces.



    Args:

        email_string: A string that represents a comma-separated list of valid emails.



    Returns:

        A list of cleaned emails.

    """

    emails = email_string.split(',')

    cleaned_emails = [email.strip() for email in emails]

    return cleaned_emails

