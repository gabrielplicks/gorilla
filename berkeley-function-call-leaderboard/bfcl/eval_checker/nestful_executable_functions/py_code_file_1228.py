from typing import Dict, Union



dpkg_architectures: Dict[str, str] = {

    'i386': 'i386',

    'amd64': 'amd64',

    'armel': 'armel',

    'armhf': 'armhf',

    'arm64': 'arm64',

    'ppc64el': 'ppc64el',

    's390x': 's390x',

    'riscv64': 'riscv64'

}



def convert_arch(arch: str) -> Union[str, None]:

    """Converts the processor architecture to the equivalent Debian package (DPKG) architecture.



    Args:

        arch: The processor architecture.



    Returns:

        The corresponding DPKG architecture string. If no mapping exists, the original `arch` string is returned.

    """

    return dpkg_architectures.get(arch, arch)

