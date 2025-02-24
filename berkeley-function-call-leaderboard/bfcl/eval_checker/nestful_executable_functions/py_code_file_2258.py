from typing import Dict



def create_user_dummy_data(user_id: int, name: str) -> Dict[str, str]:

    """Creates a dictionary with dummy data for a user with the given user ID and name.



    Args:

        user_id: The given user ID.

        name: The given user name.



    Returns:

        A dictionary with the following key-value pairs:

        - `user_id`: The given user ID.

        - `name`: The given user name.

        - `age`: A dummy age value.

        - `address`: A dummy address.

        - `email`: A dummy email address.

    """

    return {

        'user_id': str(user_id),

        'name': name,

        'age': '25',

        'address': '123 Main Street, New York, NY',

        'email': ''

    }

