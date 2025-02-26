from typing import Tuple



def create_airtable_url(app_id: str, form_id: str) -> str:

    """Creates an Airtable URL with the given application id and form id.



    Args:

        app_id: The application id.

        form_id: The form id.



    Returns:

        The Airtable URL.

    """

    template = "https://airtable.com/app/{app_id}/api/block/{form_id}/endpoint"

    return template.format(app_id=app_id, form_id=form_id)

