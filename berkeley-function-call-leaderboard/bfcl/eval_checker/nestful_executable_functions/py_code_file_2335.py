from typing import List, Dict



def calculate_total_amount(records: List[Dict[str, float]]) -> List[Dict[str, float]]:

    """Calculates the total amount for each record in a list of records.



    Args:

        records: A list of records containing a `price` field and a `quantity` field.



    Returns:

        A list of records with the `total_amount` field added, where `total_amount` is equal to `price * quantity`.

    """

    result = []

    for record in records:

        total_amount = record['price'] * record['quantity']

        result.append({**record, 'total_amount': total_amount})

    return result

