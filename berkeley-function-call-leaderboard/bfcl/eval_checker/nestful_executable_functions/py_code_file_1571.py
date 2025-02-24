from typing import Dict, List



def get_cards_from_deck(deck: Dict, headers: List[str]) -> List[Dict]:

    """Retrieves a list of cards from a deck given a list of headers.



    Args:

        deck: The deck of cards as a JSON object.

        headers: A list of headers to retrieve cards from the deck.



    Returns:

        A list of cards retrieved from the deck.

    """

    cards = []

    for header in headers:

        card = deck.get(header, {})

        cards.append(card)

    return cards

