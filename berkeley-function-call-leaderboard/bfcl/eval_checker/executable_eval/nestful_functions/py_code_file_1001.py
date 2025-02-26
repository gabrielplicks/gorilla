from typing import Union



def create_service_account(company_name: Union[str, bytes]) -> str:

    """Creates a service account name for a company.

    Args:

        company_name: The name of the company.

    """

    service_account_name = company_name.replace(" ", "_") + "_service"

    return service_account_name

