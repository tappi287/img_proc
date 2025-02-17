from typing import Sequence, Tuple

import cv2
import numpy as np
from psd_tools.api.layers import Layer
from psd_tools.constants import Tag
from psd_tools.psd.tagged_blocks import TaggedBlock
from psd_tools.psd.vector import VectorMaskSetting, Path, InitialFillRule, PathFillRule, ClosedPath, ClosedKnotLinked


def create_psd_vector_path(contours: Sequence[np.ndarray], image_size: Tuple[int, int]) -> Path:
    # -- Init data structure
    width, height = image_size[0] + 1, image_size[1] + 1
    vector_paths = Path()
    initial_fill_rule = InitialFillRule()
    initial_fill_rule.value = 0
    path_fill_rule = PathFillRule()
    vector_paths.append(path_fill_rule)
    vector_paths.append(initial_fill_rule)

    def _clip(_vec) -> float:
        return max(0.0, min(_vec, 1.0))

    # -- Convert each contour to a path
    for contour in contours:
        closed_path = ClosedPath()

        for point in contour:
            vec_x, vec_y = _clip(point[0][0] / width), _clip(point[0][1] / height)

            closed_path.append(
                ClosedKnotLinked(preceding=(vec_y, vec_x), anchor=(vec_y, vec_x), leaving=(vec_y, vec_x))
            )

        vector_paths.append(closed_path)

    return vector_paths


def create_closed_vector_mask_data(contours: Sequence[np.ndarray], image_size: Tuple[int, int]) -> VectorMaskSetting:
    return VectorMaskSetting(version=3, flags=0, path=create_psd_vector_path(contours, image_size))


def add_vector_mask_to_layer(layer: Layer, vector_mask_data: VectorMaskSetting):
    layer.tagged_blocks[Tag.VECTOR_MASK_SETTING1] = TaggedBlock(key=Tag.VECTOR_MASK_SETTING1, data=vector_mask_data)


def get_contours_from_mask(mask: np.ndarray, threshold=127, maxval=255) -> Sequence[np.ndarray]:
    if mask.dtype == np.float32:
        _mask = np.clip(255. * mask, 0, 255).astype(np.uint8)
    else:
        _mask = mask

    _, binary_mask = cv2.threshold(_mask, threshold, maxval, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours
