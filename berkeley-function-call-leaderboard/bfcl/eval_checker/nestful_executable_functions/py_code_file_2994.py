from typing import Optional



def delete_object(client: Optional[object], object_id: int) -> bool:

    """Deletes the corresponding object from a remote server via a REST API.

    Args:

        client: The client object used to access the server.

        object_id: The ID of the object to be deleted.

    Raises:

        Exception: If the client object is not provided.

    Returns:

        True if the deletion was successful, False otherwise.

    """

    if client is None:

        raise Exception("client object not provided")



    if object_id == 0:

        return False



    response = client.delete(f"https://server.com/objects/{object_id}")



    return response.ok

