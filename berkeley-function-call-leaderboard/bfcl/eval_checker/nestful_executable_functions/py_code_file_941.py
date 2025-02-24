from typing import Optional



class LinkedListNode:

    def __init__(self, value):

        self.value = value

        self.next = None



def find_middle_node(head: LinkedListNode) -> Optional[LinkedListNode]:

    """Finds the middle node of a linked list.



    Args:

        head: The head node of the linked list.



    Returns:

        The middle node of the linked list. If the linked list is empty or has only one node,

        returns None.



    Raises:

        TypeError: If the input is not a linked list node.

    """

    if not isinstance(head, LinkedListNode):

        raise TypeError("Input must be a linked list node")



    fast = head

    slow = head

    while fast and fast.next:

        fast = fast.next.next

        slow = slow.next

    return slow

