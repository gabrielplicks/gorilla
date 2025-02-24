from typing import Dict, Optional, Tuple



def parse_raw_transaction(raw_transaction: Dict) -> Optional[Tuple[float, str]]:

    """Parses a raw transaction object and returns a tuple containing the transaction amount and the recipient address.



    Args:

        raw_transaction: A dictionary representing a raw transaction object.



    Returns:

        A tuple containing the transaction amount and the recipient address, or None if the transaction does not have the appropriate fields.

    """

    if 'amount' not in raw_transaction or 'recipient' not in raw_transaction:

        return None, None



    amount = raw_transaction['amount']

    recipient = raw_transaction['recipient']



    return amount, recipient

