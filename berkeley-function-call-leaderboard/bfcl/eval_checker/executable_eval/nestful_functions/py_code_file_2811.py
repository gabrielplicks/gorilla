from typing import List, Dict



def find_instruction_with_opcode(instructions: List[Dict[str, str]], opcode: str) -> int:

    """Finds the address of the first instruction in the list that has a specific opcode.



    Args:

        instructions: A list of instructions. Each instruction is a dictionary with an "opcode" key.

        opcode: The desired opcode.



    Returns:

        The address of the first instruction in the list that has the desired opcode. If there are no instructions with the given opcode, returns -1. If the list is empty, returns -1.

    """

    for address, instruction in enumerate(instructions):

        if instruction["opcode"] == opcode:

            return address

    return -1

