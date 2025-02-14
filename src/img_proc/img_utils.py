import logging
from pathlib import Path
from typing import Optional

try:
    import OpenImageIO
    import numpy as np
except ImportError:
    pass

from .colorspace import srgbTF


def open_image(input_file: Path):
    """ Open an image file with OpenImageIO

    :param input_file: Image file to open
    :return: An OpenImageIO ImageInput object
    :rtype: Optional[OpenImageIO.ImageInput]
    """
    img_input = OpenImageIO.ImageInput.open(input_file.as_posix())

    if img_input is None:
        logging.error('Error reading image: %s', OpenImageIO.geterror())
        return

    return img_input


def open_image_with_params(img_file: Path, img_format: str = '', num_channels: int = None) -> Optional[np.array]:
    img_input = OpenImageIO.ImageInput.open(img_file.as_posix())

    if img_input is None:
        logging.error('Error reading image: %s', OpenImageIO.geterror())
        return

    spec = img_input.spec()

    # Read out image data as numpy array
    img = img_input.read_image(0, num_channels or spec.nchannels, format=img_format)
    img_input.close()

    return img


def write_image(output_file, img_spec, pixels) -> bool:
    """ Write an image at location output_file with specs img_spec with content pixels

    :param output_file: file location to write to
    :type output_file: Path
    :param img_spec: Image parameters
    :type img_spec: `class OpenImageIO.ImageSpec`
    :param pixels: Image content as numpy array
    :type pixels: `class numpy.ndarray'
    :return: True if successful
    :rtype: bool
    """
    img_out = OpenImageIO.ImageOutput.create(output_file.as_posix())
    if not img_out:
        logging.error('Error creating oiio image output:\n%s', OpenImageIO.geterror())
        return False

    result = img_out.open(output_file.as_posix(), img_spec)
    if result:
        try:
            img_out.write_image(pixels)
        except Exception as e:
            logging.error('Error writing Image file: %s', e)
            return False
    else:
        logging.error('Could not open image file for writing: %s: %s', output_file.name, img_out.geterror())
        return False

    img_out.close()
    return True


def convert_uint8_to_float(rgb):
    return np.double(rgb / 255)


def convert_float_to_uint8(rgb):
    return np.uint8(rgb * 255)


def split_image_channels_rgba(pixels):
    """ Split RGBA pixel array into RGB and A array

    :param pixels: Image content
    :type pixels: `class numpy.ndarray`
    :return: A tuple containing (RGB, A) numpy arrays
    :rtype: Tuple[`class numpy.ndarray`, `class numpy.ndarray`]
    """
    height, width, num_channels = pixels.shape

    if num_channels == 3:
        return pixels, np.zeros((height, width, 1), dtype=pixels.dtype)

    rgb = np.zeros((height, width, 3), dtype=pixels.dtype)

    r, g, b, a = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], pixels[:, :, 3]
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb, a


def un_premultiply(rgb, a):
    """ Divide every color pixel with its alpha value commonly known as un-premultiply

    :param pixels: Image RGB content in target color space
    :type pixels: `class numpy.ndarray`
    :return: transformed image content
    :rtype: `class numpy.ndarray`
    """
    return np.dstack((rgb[:, :, 0] / a, rgb[:, :, 1] / a, rgb[:, :, 2] / a))


def transform_linear_to_srgb(pixels):
    """ Transform linear colors to colors with sRGB curve applied

    :param pixels: Image content in linear color space
    :type pixels: `class numpy.ndarray`
    :return: transformed image content
    :rtype: `class numpy.ndarray`
    """
    height, width, _ = pixels.shape

    # -- Split into RGB and Alpha
    rgb, a = split_image_channels_rgba(pixels)

    # -- Apply color transform
    #    Linear to sRGB
    rgb = srgbTF(rgb, (height, width, 3), 3)
    rgba = np.dstack((rgb, a))

    return rgba
