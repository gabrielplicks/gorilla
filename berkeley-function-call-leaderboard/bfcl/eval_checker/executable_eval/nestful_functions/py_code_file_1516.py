from typing import List



def cut_cards(cards: List[int], cut_position: int) -> List[int]:

    """Simulates the cutting of cards in a deck.



    The function takes a deck of cards represented as an array of integers, where the first element is the top card of the deck, and a cut position, and returns the new order of the deck after cutting. The function performs modular arithmetic to calculate the new position of each card after the cut.



    Args:

        cards: The deck of cards represented as an array of integers.

        cut_position: The position where the deck is cut.



    Returns:

        The new order of the deck after cutting.

    """

    new_deck = []

    for i in range(cut_position, len(cards)):

        new_deck.append(cards[i])

    for i in range(cut_position):

        new_deck.append(cards[i])

    return new_deck

