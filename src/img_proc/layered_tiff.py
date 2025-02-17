import logging
from pathlib import Path
from typing import Dict, Optional, List, Sequence

import imagecodecs
import numpy as np
import psdtags
import tifffile


def create_tiff_psd_layers(tif_layers: Dict[str, "np.ndarray"]):
    psd_layers = list()
    psd_channel_idx = {
        0: psdtags.PsdChannelId.CHANNEL0,
        1: psdtags.PsdChannelId.CHANNEL1,
        2: psdtags.PsdChannelId.CHANNEL2,
        3: psdtags.PsdChannelId.TRANSPARENCY_MASK,
    }

    for idx, (name, pixels) in enumerate(tif_layers.items()):
        height, width, channel_num = pixels.shape

        # Create channels for this layer
        channels = list()
        for n in range(channel_num):
            channels.append(
                psdtags.PsdChannel(
                    channelid=psd_channel_idx[n],
                    compression=psdtags.PsdCompressionType.ZIP_PREDICTED,
                    data=pixels[..., n],
                ),
            )

        # Create psd layer
        psd_layers.append(
            psdtags.PsdLayer(
                name=name,
                rectangle=psdtags.PsdRectangle(0, 0, height, width),
                channels=channels,
                info=[
                    psdtags.PsdString(psdtags.PsdKey.UNICODE_LAYER_NAME, name),
                ],
                mask=psdtags.PsdLayerMask(),
                opacity=255,
                blendmode=psdtags.PsdBlendMode.NORMAL,
                blending_ranges=(),
                clipping=psdtags.PsdClippingType.BASE,
                flags=psdtags.PsdLayerFlag.PHOTOSHOP5,
            )
        )
    return psd_layers


def create_image_source_data(
    name: str = None, psd_layers: List["psdtags.PsdLayer"] = None, psd_format: "psdtags.PsdFormat" = None
):
    return psdtags.TiffImageSourceData(
        # --
        # name
        name=name or "Layered_Tiff",
        # --
        # psd_format
        psdformat=psd_format or psdtags.PsdFormat.LE32BIT,
        # --
        usermask=psdtags.PsdUserMask(),
        layers=psdtags.PsdLayers(
            key=psdtags.PsdKey.LAYER,
            has_transparency=False,
            # --
            # psd_layers
            layers=psd_layers or list(),
            # --
        ),
        info=[
            psdtags.PsdEmpty(psdtags.PsdKey.PATTERNS_3),
            psdtags.PsdFilterMask(
                colorspace=psdtags.PsdColorSpaceType.RGB,
                components=(65535, 0, 0, 0),
                opacity=50,
            ),
        ],
    )


def write_layered_tif_file(
    out_file: Path,
    main_image: "np.ndarray" = None,
    photoshop_image_source_data: "psdtags.TiffImageSourceData" = None,
    photoshop_image_resource_data: "psdtags.TiffImageResources" = None,
    tif_compression: Optional[str] = "ADOBE_DEFLATE",
    shape: Sequence[int] = None,
    dtype: np.dtype = None,
):
    """ Write a tif file containing Photoshop layers

    :param out_file: path to write too
    :param main_image: regular tif image_data
    :param photoshop_image_source_data: Photoshop image source data
    :param photoshop_image_resource_data: Photoshop image resource data
    :param tif_compression: Compression for the regular tif image
    :param shape: Optional numpy-like shape if main_image is omitted
    :param dtype: Optional numpy dtype if main_image is omitted
    :return:
    """
    # -- Construct Tiff Tags
    tif_extra_tags =[
        # InterColorProfile tag
        (34675, 7, None, imagecodecs.cms_profile("srgb"), True),
    ]

    if photoshop_image_source_data is not None:
        # ImageSourceData tag; use multiple threads to compress channels
        tif_extra_tags.append(
            photoshop_image_source_data.tifftag(maxworkers=6)
        )
    if photoshop_image_resource_data is not None:
        tif_extra_tags.append(photoshop_image_resource_data.tifftag())

    if main_image is None:
        tif_compression = None
        if shape is None or dtype is None:
            raise RuntimeError("You need to provide shape and dtype if no main_image data provided.")

    return tifffile.imwrite(
        out_file,
        # write composite as main TIFF image, accessible to regular TIF readers
        main_image,
        photometric="rgb",
        compression=tif_compression,
        # 72 dpi resolution
        resolution=((720000, 10000), (720000, 10000)),
        resolutionunit="inch",
        # do not write tifffile specific metadata
        metadata=None,
        # write layers and sRGB profile as extra tags
        extratags=tif_extra_tags,
        # in case main_image is None, set shape and dtype
        shape=shape, dtype=dtype
    )


def create_layered_tif_file(
    out_file: Path,
    tif_layers: Dict[str, "np.ndarray"],
    main_image: "np.ndarray",
    tif_compression: Optional[str] = "ADOBE_DEFLATE",
) -> bool:

    psd_layers = create_tiff_psd_layers(tif_layers)
    image_source_data = create_image_source_data(out_file.name, psd_layers)

    try:
        write_layered_tif_file(out_file, main_image, image_source_data, tif_compression=tif_compression)
        return True
    except Exception as err:
        logging.error(f"Error creating layered tiff file: {err}")
        return False
