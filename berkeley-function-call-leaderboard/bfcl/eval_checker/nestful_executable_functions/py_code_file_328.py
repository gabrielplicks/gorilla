from typing import Any



def object_id(obj: Any) -> str:

    """Returns a string representation of the object's identity.

    The string follows the format `"<Class-Name @ 0x<Address>>"`,

    where `<Class-Name>` is the name of the class of the object and

    `<Address>` is the hexadecimal address of the object.

    Args:

        obj: The object to get the identity of.

    """

    class_name = obj.__class__.__name__

    address = hex(id(obj))

    return f"<{class_name} @ {address}>"

