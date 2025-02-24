from typing import Optional



class User:

    name: str



class Resource:

    owner: str



def is_authorized(user: Optional[User], resource: Optional[Resource]) -> bool:

    """Checks whether a user is authorized to access a resource.

    Args:

        user: The user object.

        resource: The resource object.

    """

    if user is None or resource is None:

        return False

    if user.name == resource.owner:

        return True

    else:

        return False

