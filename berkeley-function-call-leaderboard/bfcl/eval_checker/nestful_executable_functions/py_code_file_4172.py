from typing import Optional



class Node:

    def __init__(self, data):

        self.data = data

        self.next: Optional[Node] = None



def has_loop(head: Optional[Node]) -> bool:

    """Checks if a linked list has a loop.



    Args:

        head: The head node of the linked list.



    Returns:

        A boolean indicating whether the linked list contains a loop.

    """

    slow = head

    fast = head



    while fast and fast.next:

        slow = slow.next

        fast = fast.next.next

        if slow == fast:

            return True



    return False

