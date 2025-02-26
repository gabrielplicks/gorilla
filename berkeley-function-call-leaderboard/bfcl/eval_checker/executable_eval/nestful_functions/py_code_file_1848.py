from typing import List



class Node:

    def __init__(self, value: int):

        self.value = value

        self.next: Node = None



def create_cyclic_graph(integers: List[int]) -> Node:

    """Creates a cyclic graph data structure from a nested list of integers.



    Args:

        integers: A list of integers.



    Returns:

        The root node of the graph.

    """

    root = Node(integers[0])

    current = root

    for value in integers[1:]:

        current.next = Node(value)

        current = current.next

    current.next = root

    return root

