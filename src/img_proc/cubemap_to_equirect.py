import logging
import re

from pathlib import Path
from typing import List, Optional, Callable

try:
    import OpenImageIO
    import numpy as np
except ImportError:
    pass

from . import img_utils
from .colorspace import srgbTF
from .equirectangular import build_cube_map, cube_to_equirectangular

FRAME_PATTERN = re.compile(r"\d{3,4}$")


def cubemap_images_to_equirects(img_dir: Path, output_dir: Optional[Path] = None, output_extension='.png',
                                resolution_multiplier: int = 2, use_alpha: bool = True, mode: str = "nearest",
                                progress_callback: Optional[Callable] = None, img_files: List[Path] = None,
                                is_beauty=False) -> List[Path]:
    out_dir = img_dir.parent / 'equirects_output'
    if output_dir is not None:
        out_dir = output_dir
    out_dir.mkdir(exist_ok=True)

    # -- Collect Images with the same name and frame index from 0 to 6
    cube_images = dict()

    for file in img_files or img_dir.iterdir():
        if not file.is_file():
            continue

        m = re.search(FRAME_PATTERN, file.stem)
        if not m:
            logging.info(f'Skipping file without frame index: {file}')
            continue
        frame_index = m.group()
        img_name = file.stem
        if img_name.endswith(frame_index):
            img_name = img_name[:-len(frame_index)]

        if img_name not in cube_images:
            cube_images[img_name] = list()
        cube_images[img_name].append(file)

    # -- Process images
    cube_images = {k: v for k, v in cube_images.items() if len(v) == 6}
    progress_chunk = 100 / 7 / max(1, len(cube_images.keys()))

    result_images = list()
    for img_name, images in cube_images.items():
        logging.info(f'Processing image to equirectangular map: {img_name}')
        result_images.append(
            cubemap_images_to_equirectangular(images, out_dir, output_extension, resolution_multiplier, use_alpha, mode,
                                              progress_chunk, progress_callback, is_beauty))
    if progress_callback:
        progress_callback(100.0)

    return result_images


def cubemap_images_to_equirectangular(images: List[Path], out_dir: Path, output_extension='.jpg',
                                      resolution_multiplier: int = 2, use_alpha: bool = True, mode: str = "nearest",
                                      progress_num: Optional[float] = None,
                                      progress_callback: Optional[Callable] = None, is_beauty=False) -> Path:
    eq_height = 0
    eq_width = 0
    cube_images = []
    num_channels = 4 if use_alpha else 3

    out_dir.mkdir(exist_ok=True)
    out_name = 'output'

    for img_file in sorted(images):
        m = re.search(FRAME_PATTERN, img_file.stem)
        frame_index = int(m.group())

        oiio_img = img_utils.open_image_with_params(img_file, 'float', num_channels)
        width = oiio_img.shape[1]

        if not eq_width:
            eq_width = round(width * resolution_multiplier)
            eq_height = round(eq_width / 2)
        if frame_index == 1:
            out_name = img_file.stem

        cube_images.append(oiio_img)
        if progress_callback and progress_num is not None:
            progress_callback(progress_num)

    cube_map = build_cube_map(cube_images)
    out_img = cube_to_equirectangular(cube_map, (eq_height, eq_width), interpolation=mode)

    if progress_callback and progress_num is not None:
        progress_callback(progress_num)

    # -- Write output image
    height, width, c = out_img.shape
    img_spec = OpenImageIO.ImageSpec()

    # -- Apply beauty conversion
    #    set PNG specific attributes
    #    linear-to-sRGB
    #    un-premultiply
    if is_beauty and output_extension.casefold() == '.png':
        # PNG dictates that RGBA data is always unassociated
        # (i.e., the color channels are not already premultiplied by alpha)
        img_spec["oiio:UnassociatedAlpha"] = 1
        # If nonzero and writing UINT8 values to the file from a source buffer of higher bit depth,
        # will add a small amount of random dither to combat the appearance of banding.
        img_spec["oiio:dither"] = 1

        rgb, a = img_utils.split_image_channels_rgba(out_img)

        # Un-premultiply
        rgb = img_utils.un_premultiply(rgb, a)
        # Linear to sRGB
        rgb = srgbTF(rgb, (height, width, 3), 3)

        out_img = np.dstack((rgb, a))

    # -- Write output image
    img_spec.width, img_spec.height, img_spec.nchannels = width, height, c
    output_image_path = out_dir / f'{out_name}{output_extension}'
    img_utils.write_image(output_image_path, img_spec, out_img)
    return output_image_path
