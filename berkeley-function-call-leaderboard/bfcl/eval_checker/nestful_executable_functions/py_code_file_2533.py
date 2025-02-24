from typing import Dict



def update_deletion_attributes(obj: Dict, deleted_at: float, user_id: int) -> Dict:

    """Updates the is_deleted, deleted_at, and deleted_by attributes of an object.



    Args:

        obj: The object to update.

        deleted_at: The timestamp of deletion.

        user_id: The ID of the user who deleted the object.



    Returns:

        The updated object.

    """

    obj["is_deleted"] = True

    obj["deleted_at"] = deleted_at

    obj["deleted_by"] = user_id

    return obj

