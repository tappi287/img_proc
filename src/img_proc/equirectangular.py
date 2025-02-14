import math
import numpy as np

from typing import List, Tuple, Union

NUMBA_EXISTS = True
try:
    from numba import njit, prange
except ImportError:
    NUMBA_EXISTS = False


if NUMBA_EXISTS:
    @njit(cache=True, fastmath=True)
    def _clip_coordinates(coordinates: Tuple[int, int], face_size: int)\
            -> Tuple[int, int]:
        """
        Clips coordinates at the edges of the cube map to the closest face.

        :param coordinates: The coordinates to clip.
        :type coordinates: Tuple[int, int]
        :param face_size: The size of one cube face.
        :type face_size: int
        :return: The clipped coordinates.
        :rtype: Tuple[int, int]
        """

        y, x = coordinates
        y = y % (3 * face_size)
        x = x % (4 * face_size)

        if y >= 0 and y < face_size:
            if x >= 0 and x < face_size:
                if x > y:
                    # Clip to y+
                    return y, face_size
                else:
                    # Clip to x-
                    return face_size, x

            elif x >= 2 * face_size and x < 3 * face_size:
                if y < 3 * face_size - x:
                    # Clip to y+
                    return y, 2 * face_size - 1
                else:
                    # Clip to x+
                    return face_size, x

            elif x >= 3 * face_size and x < 4 * face_size:
                # Clip to z-
                return face_size, x

        elif y >= 2 * face_size and y < 3 * face_size:
            if x >= 0 and x < face_size:
                if y - 2 * face_size < face_size - x:
                    # Clip to x-
                    return y, 2 * face_size - 1
                else:
                    # Clip to y-
                    return face_size, x

            elif x >= 2 * face_size and x < 3 * face_size:
                if x - 2 * face_size > y - 2 * face_size:
                    # Clip to x+
                    return 2 * face_size - 1, x
                else:
                    # Clip to y-
                    return y, 2 * face_size - 1

            elif x >= 3 * face_size and x < 4 * face_size:
                # Clip to z-
                return 2 * face_size - 1, x

        return y, x


    @njit(cache=True, fastmath=True)
    def _mirror_coordinates(coordinates: Tuple[int, int], face_size: int)\
            -> Tuple[int, int]:
        """
        Mirrors coordinates at the edges of the cube map to the matching face.

        :param coordinates: The coordinates to mirror.
        :type coordinates: Tuple[int, int]
        :param face_size: The size of one cube face.
        :type face_size: int
        :return: The mirrored coordinates.
        :rtype: Tuple[int, int]
        """

        y, x = coordinates
        y = y % (3 * face_size)
        x = x % (4 * face_size)

        if y >= 0 and y < face_size:
            if x >= 0 and x < face_size:
                if x > y:
                    # Mirror to x-
                    return face_size + (face_size - (x + 1)), y
                else:
                    # Mirror to y+
                    return x, face_size + (face_size - (y + 1))

            elif x >= 2 * face_size and x < 3 * face_size:
                if y < 3 * face_size - x:
                    # Mirror to x+
                    return x - face_size, 2 * face_size + (face_size - (y + 1))
                else:
                    # Mirror to y+
                    return face_size - ((x - 2 * face_size) + 1), face_size + y

            elif x >= 3 * face_size and x < 4 * face_size:
                # Mirror to y+
                return (face_size - (y + 1),
                        (2 * face_size) - (x + 1 - (3 * face_size)))

        elif y >= 2 * face_size and y < 3 * face_size:
            if x >= 0 and x < face_size:
                if y - 2 * face_size < face_size - x:
                    # Mirror to y-
                    return (2 * face_size + (face_size - (x + 1)),
                            face_size + (y - 2 * face_size))
                else:
                    # Mirror to x-
                    return (2 * face_size - (face_size - x),
                            face_size - ((y + 1) - 2 * face_size))

            elif x >= 2 * face_size and x < 3 * face_size:
                if x - 2 * face_size > y - 2 * face_size:
                    # Mirror to y-
                    return x, 2 * face_size - ((y + 1) - 2 * face_size)
                else:
                    # Mirror to x+
                    return 2 * face_size - ((x + 1) - 2 * face_size), y

            elif x >= 3 * face_size and x < 4 * face_size:
                # Mirror to y-
                return (3 * face_size - ((y + 1) - 2 * face_size),
                        2 * face_size - ((x + 1) - 3 * face_size))

        return y, x


    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _cube_to_equirectangular_nearest(cube_array: np.ndarray,
                                         map_size: Tuple[int, int]) -> np.ndarray:
        """
        Turns a cube map into an equirectangular map. No interpolation.

        :param cube_array: The cube image.
        :type cube_array: np.ndarray
        :param map_size: The new size of the array in y and x coordinates in that
                         order.
        :type map_size: Tuple[int, int]
        :return: The equirectangular map.
        :rtype: np.ndarray
        """

        # Normalized output coordinates and polar coordinates.
        u, v = 0., 0.
        phi, theta = 0., 0.

        # Pixel positions and offsets.
        x_in, y_in, x_offset, y_offset = 0, 0, 0, 0

        # Set up dimensions and output.
        cube_map_height, cube_map_width, channels = cube_array.shape
        cube_face_height = cube_map_height // 3
        cube_face_width = cube_map_width // 4

        map_y_size, map_x_size = map_size
        map_y_max, map_x_max = map_y_size - 1, map_x_size - 1
        array_out = np.zeros((map_y_size, map_x_size, channels),
                             dtype=cube_array.dtype)

        for y_out in prange(map_y_size):
            v = 1. - (y_out / map_y_max)
            theta = v * np.pi
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            y_abs = abs(cos_theta)
            for x_out in prange(map_x_size):
                u = x_out / map_x_max
                phi = u * 2. * np.pi

                # Unit vector.
                x = np.sin(phi) * sin_theta * -1.
                y = cos_theta
                z = np.cos(phi) * sin_theta * -1.
                a = max(abs(x), y_abs, abs(z))

                # Vector parallel to the unit vector that lies on one of the cube
                # faces.
                xa, ya, za = x / a, y / a, z / a

                # Get pixels and offsets.
                if x == a:
                    x_in = (((za + 1.) / 2.) - 1.) * cube_face_width
                    x_offset = 2 * cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif x == -a:
                    x_in = ((za + 1.) / 2.) * cube_face_width
                    x_offset = 0
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif y == a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = (((za + 1.) / 2.) - 1.) * cube_face_height
                    y_offset = 2 * cube_face_height
                elif y == -a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = ((za + 1.) / 2.) * cube_face_height
                    y_offset = 0
                elif z == a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif z == -a:
                    x_in = (((xa + 1.) / 2.) - 1.) * cube_face_width
                    x_offset = 3 * cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height

                x_in = abs(int(x_in)) + x_offset
                y_in = abs(int(y_in)) + y_offset

                for c in prange(channels):
                    array_out[y_out, x_out, c] = cube_array[y_in, x_in, c]

        return array_out


    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _cube_to_equirectangular_bilinear(cube_array: np.ndarray,
                                          map_size: Tuple[int, int]) -> np.ndarray:
        """
        Turns a cube map into an equirectangular map. Interpolates bilinearly.

        :param cube_array: The cube images.
        :type cube_array: List[np.ndarray]
        :param map_size: The new size of the array in y and x coordinates in that
                         order.
        :type map_size: Tuple[int, int]
        :return: The resized array.
        :rtype: np.ndarray
        """

        # Normalized output coordinates and polar coordinates.
        u, v = 0., 0.
        phi, theta = 0., 0.

        # Pixel positions and offsets.
        x_in, y_in, x_offset, y_offset = 0, 0, 0, 0

        # Set up dimensions and output.
        cube_map_height, cube_map_width, channels = cube_array.shape
        cube_face_height = cube_map_height // 3
        cube_face_width = cube_map_width // 4

        map_y_size, map_x_size = map_size
        map_y_max, map_x_max = map_y_size - 1, map_x_size - 1
        array_out = np.zeros((map_y_size, map_x_size, channels),
                             dtype=cube_array.dtype)

        for y_out in prange(map_y_size):
            v = 1. - (y_out / map_y_max)
            theta = v * np.pi
            sin_theta = np.sin(theta)
            y = np.cos(theta)
            y_abs = abs(y)
            for x_out in prange(map_x_size):
                u = x_out / map_x_max
                phi = u * 2. * np.pi

                # Unit vector.
                x = np.sin(phi) * sin_theta * -1.
                # y = np.cos(theta)
                z = np.cos(phi) * sin_theta * -1.
                a = max(abs(x), y_abs, abs(z))

                # Vector parallel to the unit vector that lies on one of the cube
                # faces.
                xa, ya, za = x / a, y / a, z / a

                # Get pixels and offsets.
                if x == a:
                    x_in = (((za + 1.) / 2.) - 1.) * cube_face_width
                    x_offset = 2 * cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif x == -a:
                    x_in = ((za + 1.) / 2.) * cube_face_width
                    x_offset = 0
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif y == a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = (((za + 1.) / 2.) - 1.) * cube_face_height
                    y_offset = 2 * cube_face_height
                elif y == -a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = ((za + 1.) / 2.) * cube_face_height
                    y_offset = 0
                elif z == a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif z == -a:
                    x_in = (((xa + 1.) / 2.) - 1.) * cube_face_width
                    x_offset = 3 * cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height

                # Sub-pixel position.
                x_in = max(0., abs(x_in) - .5)
                y_in = max(0., abs(y_in) - .5)
                xo_in = x_in + x_offset
                yo_in = y_in + y_offset

                x_l = int(xo_in)
                x_r = math.ceil(xo_in)
                y_t = int(yo_in)
                y_b = math.ceil(yo_in)

                # Mirror coordinates for interpolation.
                q11_c = _mirror_coordinates((y_b, x_l), cube_face_width)
                q12_c = (y_t, x_l)
                q21_c = _mirror_coordinates((y_b, x_r), cube_face_width)
                q22_c = _mirror_coordinates((y_t, x_r), cube_face_width)

                x_fac = xo_in - x_l
                y_fac = yo_in - y_t

                for c in prange(channels):
                    # For the terms see:
                    # https://en.wikipedia.org/wiki/Bilinear_interpolation

                    # Sample surrounding pixels.
                    q11 = cube_array[q11_c[0], q11_c[1], c]
                    q12 = cube_array[q12_c[0], q12_c[1], c]
                    q21 = cube_array[q21_c[0], q21_c[1], c]
                    q22 = cube_array[q22_c[0], q22_c[1], c]

                    # Interpolate horizontally.
                    r1 = (1 - x_fac) * q11 + x_fac * q21
                    r2 = (1 - x_fac) * q12 + x_fac * q22

                    # Interpolate vertically.
                    p = (1 - y_fac) * r2 + y_fac * r1

                    array_out[y_out, x_out, c] = p

        return array_out


    def _cube_to_equirectangular_bicubic(cube_array: np.ndarray,
                                         map_size: Tuple[int, int]) -> np.ndarray:
        """
        Turns a cube map into an equirectangular map. Interpolates bicubicly.

        :param cube_array: The cube images.
        :type cube_array: List[np.ndarray]
        :param map_size: The new size of the array in y and x coordinates in that
                         order.
        :type map_size: Tuple[int, int]
        :return: The resized array.
        :rtype: np.ndarray
        """

        # Defines upper and lower bounds for the given data type to handle spline
        # over and undershoots.
        if np.issubdtype(cube_array.dtype, np.integer):
            dmin = 0
            dmax = np.iinfo(cube_array.dtype).max
        else:
            dmin = 0.
            dmax = np.finfo(cube_array.dtype).max

        return _cube_to_equirectangular_bicubic_(cube_array, map_size, dmin, dmax)


    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _cube_to_equirectangular_bicubic_(cube_array: np.ndarray,
                                          map_size: Tuple[int, int],
                                          dmin: Union[int, np.number],
                                          dmax: Union[int, np.number]) -> np.ndarray:
        """
        Turns a cube map into an equirectangular map. Interpolates bicubicly.

        :param cube_array: The cube images.
        :type cube_array: List[np.ndarray]
        :param map_size: The new size of the array in y and x coordinates in that
                         order.
        :type map_size: Tuple[int, int]
        :param dmin: The lower bound for the given data type.
        :type dmin: Union[int, np.number]
        :param dmax: The upper bound for the given data type.
        :type dmax: Union[int, np.number]
        :return: The resized array.
        :rtype: np.ndarray
        """

        # Normalized output coordinates and polar coordinates.
        u, v = 0., 0.
        phi, theta = 0., 0.

        # Pixel positions and offsets.
        x_in, y_in, x_offset, y_offset = 0, 0, 0, 0

        # Set up dimensions and output.
        cube_map_height, cube_map_width, channels = cube_array.shape
        cube_face_height = cube_map_height // 3
        cube_face_width = cube_map_width // 4

        map_y_size, map_x_size = map_size
        map_y_max, map_x_max = map_y_size - 1, map_x_size - 1
        array_out = np.zeros((map_y_size, map_x_size, channels),
                             dtype=cube_array.dtype)

        for y_out in prange(map_y_size):
            v = 1. - (y_out / map_y_max)
            theta = v * np.pi
            sin_theta = np.sin(theta)
            y = np.cos(theta)
            y_abs = abs(y)
            for x_out in prange(map_x_size):
                u = x_out / map_x_max
                phi = u * 2. * np.pi

                # Unit vector.
                x = np.sin(phi) * sin_theta * -1.
                # y = np.cos(theta)
                z = np.cos(phi) * sin_theta * -1.
                a = max(abs(x), y_abs, abs(z))

                # Vector parallel to the unit vector that lies on one of the cube
                # faces.
                xa, ya, za = x / a, y / a, z / a

                # Get pixels and offsets.
                if x == a:
                    x_in = (((za + 1.) / 2.) - 1.) * cube_face_width
                    x_offset = 2 * cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif x == -a:
                    x_in = ((za + 1.) / 2.) * cube_face_width
                    x_offset = 0
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif y == a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = (((za + 1.) / 2.) - 1.) * cube_face_height
                    y_offset = 2 * cube_face_height
                elif y == -a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = ((za + 1.) / 2.) * cube_face_height
                    y_offset = 0
                elif z == a:
                    x_in = ((xa + 1.) / 2.) * cube_face_width
                    x_offset = cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height
                elif z == -a:
                    x_in = (((xa + 1.) / 2.) - 1.) * cube_face_width
                    x_offset = 3 * cube_face_width
                    y_in = ((ya + 1.) / 2.) * cube_face_height
                    y_offset = cube_face_height

                # Sub-pixel position.
                x_in = max(0., abs(x_in) - .5)
                y_in = max(0., abs(y_in) - .5)
                xo_in = x_in + x_offset
                yo_in = y_in + y_offset
                ax = int(xo_in)
                ay = int(yo_in)

                # Mirror coordinates for interpolation.
                a00 = _mirror_coordinates((ay - 1, ax - 1), cube_face_width)
                a01 = _mirror_coordinates((ay - 1, ax), cube_face_width)
                a02 = _mirror_coordinates((ay - 1, ax + 1), cube_face_width)
                a03 = _mirror_coordinates((ay - 1, ax + 2), cube_face_width)
                a10 = _mirror_coordinates((ay, ax - 1), cube_face_width)
                a11 = (ay, ax)
                a12 = _mirror_coordinates((ay, ax + 1), cube_face_width)
                a13 = _mirror_coordinates((ay, ax + 2), cube_face_width)
                a20 = _mirror_coordinates((ay + 1, ax - 1), cube_face_width)
                a21 = _mirror_coordinates((ay + 1, ax), cube_face_width)
                a22 = _mirror_coordinates((ay + 1, ax + 1), cube_face_width)
                a23 = _mirror_coordinates((ay + 1, ax + 2), cube_face_width)
                a30 = _mirror_coordinates((ay + 2, ax - 1), cube_face_width)
                a31 = _mirror_coordinates((ay + 2, ax), cube_face_width)
                a32 = _mirror_coordinates((ay + 2, ax + 1), cube_face_width)
                a33 = _mirror_coordinates((ay + 2, ax + 2), cube_face_width)

                x_t = xo_in - ax
                y_t = yo_in - ay

                # For the terms see the matrix at:
                # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
                x_term_0 = 0.5 * (-x_t ** 3 + 2 * x_t ** 2 - x_t)
                x_term_1 = 0.5 * (3 * x_t ** 3 - 5 * x_t ** 2 + 2)
                x_term_2 = 0.5 * (-3 * x_t ** 3 + 4 * x_t ** 2 + x_t)
                x_term_3 = 0.5 * (x_t ** 3 - x_t ** 2)

                y_term_0 = 0.5 * (-y_t ** 3 + 2 * y_t ** 2 - y_t)
                y_term_1 = 0.5 * (3 * y_t ** 3 - 5 * y_t ** 2 + 2)
                y_term_2 = 0.5 * (-3 * y_t ** 3 + 4 * y_t ** 2 + y_t)
                y_term_3 = 0.5 * (y_t ** 3 - y_t ** 2)

                for c in prange(channels):
                    # Sample surrounding pixels.
                    ca00 = cube_array[a00[0], a00[1], c]
                    ca01 = cube_array[a01[0], a01[1], c]
                    ca02 = cube_array[a02[0], a02[1], c]
                    ca03 = cube_array[a03[0], a03[1], c]
                    ca10 = cube_array[a10[0], a10[1], c]
                    ca11 = cube_array[a11[0], a11[1], c]
                    ca12 = cube_array[a12[0], a12[1], c]
                    ca13 = cube_array[a13[0], a13[1], c]
                    ca20 = cube_array[a20[0], a20[1], c]
                    ca21 = cube_array[a21[0], a21[1], c]
                    ca22 = cube_array[a22[0], a22[1], c]
                    ca23 = cube_array[a23[0], a23[1], c]
                    ca30 = cube_array[a30[0], a30[1], c]
                    ca31 = cube_array[a31[0], a31[1], c]
                    ca32 = cube_array[a32[0], a32[1], c]
                    ca33 = cube_array[a33[0], a33[1], c]

                    # Interpolate horizontally.
                    r0 = ca00 * x_term_0 + ca01 * x_term_1\
                         + ca02 * x_term_2 + ca03 * x_term_3

                    r1 = ca10 * x_term_0 + ca11 * x_term_1\
                         + ca12 * x_term_2 + ca13 * x_term_3

                    r2 = ca20 * x_term_0 + ca21 * x_term_1\
                         + ca22 * x_term_2 + ca23 * x_term_3

                    r3 = ca30 * x_term_0 + ca31 * x_term_1\
                         + ca32 * x_term_2 + ca33 * x_term_3

                    # Interpolate vertically.
                    # 1e-12 is a simple and performant fix for numerical
                    # instability. Necessary for a clean alpha channel.
                    p = r0 * y_term_0 + r1 * y_term_1\
                        + r2 * y_term_2 + r3 * y_term_3\
                        + 1e-12

                    array_out[y_out, x_out, c] = min(max(dmin, p), dmax)

        return array_out


def build_cube_map(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Expects the images in the following order.

    0001 - back   - x-
    0002 - left   - z+
    0003 - front  - x+
    0004 - right  - z-
    0005 - top    - y+
    0006 - bottom - y-
    """

    height, width, channels = arrays[0].shape
    c_width = width * 4
    c_height = height * 3

    # Initialize cube map.
    cube_map = np.zeros((c_height, c_width, channels),
                         dtype=arrays[0].dtype)

    # Set sub arrays.
    cube_map[height:2*height, 0:width] = arrays[0]
    cube_map[height:2*height, width:2*width] = arrays[1]
    cube_map[height:2*height, 2*width:3*width] = arrays[2]
    cube_map[height:2*height, 3*width:4*width] = arrays[3]
    cube_map[0:height, 2*width:3*width] = np.flip(arrays[4], (0, 1))
    cube_map[2*height:3*height, 2*width:3*width] = np.flip(arrays[5], (0, 1))
    cube_map = np.roll(cube_map, -width, axis=1)

    return cube_map


def cube_to_equirectangular(cube_array: np.ndarray,
                            map_size: Tuple[int, int],
                            interpolation: str = "nearest")\
        -> Union[np.ndarray, None]:
    """
    Turns a cube map into an equirectangular map.

    Returns the equirectangular map as a numpy array or returns None if no
    valid interpolation is provided or Numba can't be imported.
    Defaults to 'nearest' as default interpolation.

    Valid interpolations are:
    - nearest
    - bilinear
    - bicubic

    :param cube_array: The cube images.
    :type cube_array: List[np.ndarray]
    :param map_size: The new size of the array in y and x coordinates in that
                     order.
    :type map_size: Tuple[int, int]
    :param interpolation: The interpolation to use. Defaults to 'nearest'.
    :type interpolation: str
    :return: The resized array.
    :rtype: np.ndarray
    """

    if not NUMBA_EXISTS:
        return None

    if interpolation == "nearest":
        return _cube_to_equirectangular_nearest(cube_array, map_size)
    elif interpolation == "bilinear":
        return _cube_to_equirectangular_bilinear(cube_array, map_size)
    elif interpolation == "bicubic":
        return _cube_to_equirectangular_bicubic(cube_array, map_size)

    return None
