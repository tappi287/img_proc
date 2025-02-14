import logging
from pathlib import Path
from typing import Union, Dict

try:
    import OpenImageIO
    IMG_IO_AVAIL = True
except ImportError:
    IMG_IO_AVAIL = False
import numpy as np

ICC_SRGB_PROFILE = None


def get_srgb_icc_profile() -> Optional["np.ndarray"]:
    if not IMG_IO_AVAIL:
        return None

    global ICC_SRGB_PROFILE

    if ICC_SRGB_PROFILE is None:
        # -- Load ICC Profile
        with open(get_data_dir() / ICC_SRGB, "rb") as f:
            ICC_SRGB_PROFILE = np.load(f)
    return ICC_SRGB_PROFILE


def write_tif(image_pixels: "np.ndarray", output_file: Union[str, Path]):
    """Write tiff with embedded srgb profile, 32bit float"""
    if not IMG_IO_AVAIL:
        return False

    from . import img_utils

    height, width, channel_num = image_pixels.shape

    tif_spec = OpenImageIO.ImageSpec(width, height, channel_num, "float")
    tif_spec["compression"] = "zip"

    # - embed icc profile
    icc_profile = get_srgb_icc_profile()
    if icc_profile is not None:
        tif_spec.attribute("ICCProfile", OpenImageIO.TypeDesc(f"uint8[{len(icc_profile)}]"), icc_profile)

    return img_utils.write_image(Path(output_file), tif_spec, image_pixels)


def write_mask_multi_layer_tif(tif_layers: Dict[str, "np.ndarray"], output_file: Union[str, Path]) -> bool:
    filename = Path(output_file).as_posix()

    # Create the ImageOutput
    out = OpenImageIO.ImageOutput.create(filename)
    if not out:
        logging.error("Could not open image file for writing: %s: %s", filename, out.geterror())
        return False

    if not out.supports("multiimage") or not out.supports("appendsubimage"):
        logging.error(f"{filename} does not support writing multiple sub images.")
        return False

    # Prepare specs for every layer
    tif_specs = list()
    for idx, (mask_name, image_pixels) in enumerate(tif_layers.items()):
        height, width, channel_num = image_pixels.shape
        tif_spec = OpenImageIO.ImageSpec(width, height, channel_num, "float")
        tif_spec["compression"] = "lzw"
        tif_spec["Software"] = "Python"
        tif_spec["tiff:PageName"] = mask_name
        tif_spec["tiff:PageNumber"] = idx
        tif_spec["tiff:planarconfig"] = "contig"
        tif_spec["tiff:subfiletype"] = 0

        tif_specs.append(tif_spec)

    # Write layers
    out.open(filename, tif_specs)
    for idx, image_pixels in enumerate(tif_layers.values()):
        if idx > 0:
            out.open(filename, tif_specs[idx], "AppendSubimage")
        out.write_image(image_pixels)

    out.close()
    return True
