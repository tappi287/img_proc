import logging
from pathlib import Path
from typing import Union, Optional

from . import img_utils, colorspace

try:
    # Binary module may not be available in every Python Version
    import numpy as np
    import OpenImageIO

    OIIO_AVAIL = True
except ImportError:
    OIIO_AVAIL = False


def convert_exr_to_png(input_file: Union[str, Path], output_file: Union[str, Path],
                       use_dithering: bool = True, linear_to_srgb: bool = True,
                       un_premultiply: bool = True) -> bool:
    """ Convert a 16bit float linear EXR into a 8bit PNG.

    :param input_file: Input file path
    :param output_file: output file path
    :param use_dithering: Whether to add randomness to output color values to fight banding
    :param linear_to_srgb: Whether to apply a sRGB transform to the color values
    :param un_premultiply: Whether to un-premultiply color values with the alpha values.
                           Divides color values with the alpha rgb / a
    :return: True on success
    """
    if not OIIO_AVAIL:
        logging.error('OpenImageIO Python package or NumPy is not available!')
        return False

    # -- Create an OpenImageIO ImageInput pointing to input_file
    input_file = Path(input_file)
    img_input = img_utils.open_image(input_file)
    if not img_input:
        return False

    # -- Create Image Spec describing image properties
    img_spec: OpenImageIO.ImageSpec = img_input.spec()

    # PNG dictates that RGBA data is always unassociated
    # (i.e., the color channels are not already premultiplied by alpha)
    img_spec["oiio:UnassociatedAlpha"] = 1

    if use_dithering:
        # If nonzero and writing UINT8 values to the file from a source buffer of higher bit depth,
        # will add a small amount of random dither to combat the appearance of banding.
        img_spec["oiio:dither"] = 1
    else:
        img_spec["oiio:dither"] = 0

    pixels = img_input.read_image("float")

    # -- Un-Premultiply
    if un_premultiply:
        rgb, a = img_utils.split_image_channels_rgba(pixels)
        rgb = img_utils.un_premultiply(rgb, a)
        pixels = np.dstack((rgb, a))

    # -- Read image pixels as float32 and convert linear to sRgb
    if linear_to_srgb:
        pixels = img_utils.transform_linear_to_srgb(pixels)

    # -- Write output image
    try:
        write_result = img_utils.write_image(output_file, img_spec, pixels)
    except Exception as e:
        logging.error('Error writing image file: %s', e)
        write_result = False

    # -- Close input image
    img_input.close()

    return write_result


def png_to_unpremult(input_file: Union[str, Path], output_file: Union[None, str, Path] = None,
                     use_pillow: bool = True, verbose: bool = False) -> bool:
    """ Experimental method to open a 8bit PNG, un-premultiply it's color channels and write it
        back to PNG.

        :param input_file: Input file path
        :param output_file: Optional output file path, leave at None to overwrite input file path
        :param use_pillow: will use Pillow and produce results close to Nuke Un-premultiply
        :param verbose: Prints out info about a test pixel
        :return: True on success
    """
    # -- Create an OpenImageIO ImageInput pointing to input_file
    input_file, img_input, img_width, img_height = Path(input_file), None, -1, -1

    if not use_pillow:
        img_input: Optional[OpenImageIO.ImageInput] = img_utils.open_image(input_file)
        if not img_input:
            return False
        img_spec: OpenImageIO.ImageSpec = img_input.spec()
        img_spec["oiio:UnassociatedAlpha"] = 0
        img_spec["oiio:dither"] = 0
        img_spec["oiio:ColorSpace"] = "sRGB"
        img_width, img_height = img_spec.width, img_spec.height,
        if verbose:
            print('Input:', img_spec['oiio:ColorSpace'], img_spec.format)
        img_pixels = img_input.read_image("uint8")
    else:
        try:
            from PIL import Image
        except ImportError:
            logging.error('Pillow Python Package not available.')
            return False

        pil_img = Image.open(input_file)
        img_width, img_height = pil_img.width, pil_img.height
        img_pixels = np.asarray(pil_img)

    # -- Read pixels as 8bit integer values
    rgb, a = img_utils.split_image_channels_rgba(img_pixels)
    # Nuke: Red 6 , 0.0217, Oiio: 33, Pil: 41 (ssems about right)
    if verbose:
        print('Red:', rgb[266, 156, 0], 'Alpha:', a[266, 156], 'Premul:', rgb[266, 156, 0] / a[266, 156])

    # -> 8bit INT to FLOAT
    float_rgb, float_a = img_utils.convert_uint8_to_float(rgb), img_utils.convert_uint8_to_float(a)
    if verbose:
        print('8bit INT to FLOAT')
        print('Red:', float_rgb[266, 156, 0], 'Alpha:', float_a[266, 156])

    # -> Convert sRGB to Linear
    float_rgb = colorspace.srgbTF(float_rgb, (img_height, img_width, 3), 3, reverse=True)
    if verbose:
        print('Convert sRGB to Linear')
        print('Red:', float_rgb[266, 156, 0], 'Alpha:', float_a[266, 156])

    # -> Un-premultiply
    float_rgb = img_utils.un_premultiply(float_rgb, float_a)
    if verbose:
        print('Un-premultiply')
        print('Red:', float_rgb[266, 156, 0], 'Alpha:', float_a[266, 156])

    # -> Convert Linear to sRGB
    float_rgb = colorspace.srgbTF(float_rgb, (img_height, img_width, 3), 3)
    if verbose:
        print('Convert Linear to sRGB')
        print('Red:', float_rgb[266, 156, 0], 'Alpha:', float_a[266, 156])

    # -- Create uint8 RGBA pixel array
    pixels = np.dstack((img_utils.convert_float_to_uint8(float_rgb),
                        img_utils.convert_float_to_uint8(float_a)))

    # Nuke coord: x 156, y 213
    if verbose:
        print('Create uint8 RGBA pixel array')
        print('Red:', pixels[266, 156, 0], 'Alpha:', pixels[266, 156, 3])

    if not output_file:
        output_file = input_file

    if img_input:
        img_input.close()

    if not use_pillow:
        write_result = img_utils.write_image(output_file, img_spec, pixels)
    else:
        pil_img = Image.fromarray(pixels)
        try:
            pil_img.save(output_file)
            write_result = True
        except Exception as e:
            logging.error('Error writing image: %s', e)
            write_result = False

    return write_result
