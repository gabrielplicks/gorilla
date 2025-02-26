from typing import Dict



def is_senior_citizen(properties: Dict[str, any]) -> bool:

    """Checks if a person is a senior citizen based on their age and marital status.



    Args:

        properties: A dictionary containing the properties of a person, including 'age' and 'marital_status'.



    Returns:

        True if the person is a senior citizen, False otherwise.

    """

    age = properties.get('age')

    marital_status = properties.get('marital_status')



    if age >= 60:

        return True

    elif marital_status == 'married' and age >= 65:

        return True

    return False

