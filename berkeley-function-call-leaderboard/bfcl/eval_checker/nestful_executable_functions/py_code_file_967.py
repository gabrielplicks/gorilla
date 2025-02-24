import random

random.seed(42)
import string

from typing import Dict



def generate_email_message() -> Dict[str, str]:

    """Generates a random email message with a random string as the email address, a random number as the subject, and a random string as the body.



    Args:

        None



    Returns:

        A dictionary with keys 'email', 'subject', and 'body'.

    """

    random_string: str = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

    email: str = random_string + '@gmail.com'

    subject: str = str(random.randint(1, 1000))

    body: str = random_string * 2

    message: Dict[str, str] = {

        'email': email,

        'subject': subject,

        'body': body

    }



    return message

