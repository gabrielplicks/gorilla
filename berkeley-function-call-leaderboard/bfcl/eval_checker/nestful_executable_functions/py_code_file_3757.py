import json

from typing import Any



def get_card_data(review_data: str) -> str:

    """Extracts the created time, modified time, and rating status from the input card review data and returns a formatted string containing the extracted information.



    Args:

        review_data: A string of JSON-formatted data.



    Returns:

        A formatted string containing the extracted information.

    """

    data: dict[str, Any] = json.loads(review_data)

    created_time: str = data["created"]

    modified_time: str = data["modified"]

    rating_status: str = data["ratingStatus"]

    return f"Created at: {created_time}, Modified at: {modified_time}, Rating Status: {rating_status}"

