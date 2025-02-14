import logging
import re
from pathlib import Path
from typing import List, Dict

IMG_SEQ_REPLACE_PATTERN = re.compile(r"\d+$")
MISSING_FRAMES_LIMIT = 20


def get_normalized_image_name(image: Path) -> str:
    return re.sub(IMG_SEQ_REPLACE_PATTERN, "", image.stem)


def create_image_sequence_file_name(image_files: List[Path]) -> str:
    if not image_files:
        return str()

    img_name = get_normalized_image_name(image_files[0])
    img_d_start = image_files[0].stem.replace(img_name, "")
    img_d_end = image_files[-1].stem.replace(img_name, "")
    return f"{img_name}[{img_d_start}-{img_d_end}]{image_files[0].suffix}"


def get_image_sequences(image_files: List[Path]) -> Dict[str, List[Path]]:
    img_seq_map = dict()
    for image in image_files:
        normalized_img_name = get_normalized_image_name(image) + image.suffix
        if normalized_img_name not in img_seq_map:
            img_seq_map[normalized_img_name] = list()
        img_seq_map[normalized_img_name].append(image)

    # -- Collect images with same normalized name
    return {k: sorted([i for i in v], key=lambda i: i.stem) for k, v in img_seq_map.items() if len(v) > 1}


def find_missing_frames_in_image_sequence(img_seq_files: List[Path]) -> List[int]:
    if not img_seq_files:
        return list()

    img_seq_name = get_normalized_image_name(img_seq_files[0])
    frame_name = img_seq_files[0].stem.replace(img_seq_name, "")
    if not frame_name.isdigit():
        return list()
    prev_img_digit = int(frame_name)
    missing_frames = list()

    for img_seq_file in img_seq_files:
        # -- Collect missing image sequence frames
        img_digit = int(img_seq_file.stem.replace(img_seq_name, ""))
        if img_digit - prev_img_digit > 1:
            for idx, c in enumerate(range(prev_img_digit + 1, img_digit)):
                missing_frames.append(c)
                if idx >= MISSING_FRAMES_LIMIT:
                    break

        prev_img_digit = img_digit

    return missing_frames


def find_missing_frames_in_image_sequences(img_seq_map: Dict[str, List[Path]]) -> Dict[str, List[int]]:
    img_seq_missing_map = dict()
    for img_seq, img_seq_files in img_seq_map.items():
        missing_frames = find_missing_frames_in_image_sequence(img_seq_files)

        if missing_frames:
            img_seq_missing_map[img_seq] = missing_frames
            logging.info(f"Found missing frames: {missing_frames} in [{img_seq}]")

    return img_seq_missing_map
