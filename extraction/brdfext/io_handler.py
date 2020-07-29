import numpy as np
import pickle
import rawpy
import os


def read_raw(raw_path, wb):
    """
    Reads and postprocesses a RAW file
    :param raw_path: Path to RAW file
    :param wb: Perform white balance
    :return: Read RAW file and CFA pattern for each pixel in image
    """

    raw = rawpy.imread(raw_path)

    cfa_pattern = raw.raw_colors_visible

    rgb = raw.postprocess(use_camera_wb=False, use_auto_wb=False, user_wb=list(wb),
                          no_auto_bright=True, user_flip=0, bright=2)

    img = np.zeros(rgb.shape, dtype=np.uint8)
    img[:, :, 0] = rgb[:, :, 2]
    img[:, :, 1] = rgb[:, :, 1]
    img[:, :, 2] = rgb[:, :, 0]

    return [img, cfa_pattern]


def save_data(path, obj):
    """
    Writes an object to a file
    :param path: Path to file
    :param obj: Object to write to file
    :return: Void
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_data(path):
    """
    Reads an object from a file
    :param path: Path to file
    :return: Read object
    """
    if not os.path.isfile(path):
        return None

    with open(path, 'rb') as f:
        return pickle.load(f)

