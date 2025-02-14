import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import imagecodecs
    import numpy as np
    import psdtags
    import tifffile

    IMG_IO_AVAIL = True
except ImportError:
    imagecodecs = None
    np = None
    psdtags = None
    tifffile = None
    IMG_IO_AVAIL = False

from .utils.generic import exception_and_traceback


def create_layered_tif_file(
    out_file: Path,
    tif_layers: Dict[str, "np.ndarray"],
    main_image: "np.ndarray",
    tif_compression: Optional[str] = "ADOBE_DEFLATE",
) -> bool:
    if not IMG_IO_AVAIL:
        logging.fatal("Image processing is not available. Please install imagecodecs first.")
        return False

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

    image_source_data = psdtags.TiffImageSourceData(
        name="Layered_Tiff",
        psdformat=psdtags.PsdFormat.LE32BIT,
        usermask=psdtags.PsdUserMask(),
        layers=psdtags.PsdLayers(
            key=psdtags.PsdKey.LAYER,
            has_transparency=False,
            layers=psd_layers,
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

    # write a layered TIFF file
    try:
        tifffile.imwrite(
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
            extratags=[
                # ImageSourceData tag; use multiple threads to compress channels
                image_source_data.tifftag(maxworkers=6),
                # InterColorProfile tag
                (34675, 7, None, imagecodecs.cms_profile("srgb"), True),
            ],
        )
    except Exception as err:
        logging.error(f"Error creating layered tiff file: {exception_and_traceback(err)}")
        return False

    return True
