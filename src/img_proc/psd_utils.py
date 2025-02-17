import io
from typing import Union

import psd_tools.psd.vector
from psd_tools import psd

from psdtags import psdtags

PATH_DEFAULT_NAME = "Pfad 1"


def psd_tools_obj_to_bytes(psd_tools_obj: psd.BaseElement) -> bytes:
    """ Write psd_tools object to a byte buffer and retrieve as bytes """
    with io.BytesIO() as f:
        psd_tools_obj.write(f)
        f.seek(0)
        byte_data = f.read()

    return byte_data


def create_psd_tif_image_resources(name: str) -> psdtags.TiffImageResources:
    return psdtags.TiffImageResources(
        psdformat=psdtags.PsdFormat.BE32BIT,
        name=name,
        blocks=list()
    )


def create_psd_path_resource_block(name: str = None, path_data: Union[bytes, psd_tools.psd.vector.Path] = None):
    if name is None:
        name = PATH_DEFAULT_NAME
    if path_data is None:
        path_data = bytes()

    if isinstance(path_data, psd_tools.psd.vector.Path):
        path_data = psd_tools_obj_to_bytes(path_data)

    return psdtags.PsdBytesBlock(
        resourceid=psdtags.PsdResourceId.PATH_INFO, name=name, value=path_data
    )
