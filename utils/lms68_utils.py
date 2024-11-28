import numpy as np


def get_jaw(lms):
    return lms[:17]


def get_left_eye_brow(lms):
    return lms[17:22]


def get_right_eye_brow(lms):
    return lms[22:27]


def get_nose(lms):
    return lms[27:36]


def get_left_eye(lms):
    return lms[36:42]


def get_right_eye(lms):
    return lms[42:48]


def get_mouth(lms):
    return lms[48:]


def get_roi(lms):
    top_left = np.min(lms, axis=0)
    bottom_right = np.max(lms, axis=0)
    return np.concatenate([top_left, bottom_right], axis=0)


def get_center(lms):
    roi = get_roi(lms)
    return (roi[:2] + roi[2:]) / 2


def get_square_roi(lms, enlarge_ratio=1.0):
    roi = get_roi(lms)
    top_left = roi[:2]
    bottom_right = roi[2:]
    center = (top_left + bottom_right) / 2
    half_len = np.max(center - top_left)
    half_len *= enlarge_ratio
    return np.concatenate([center - half_len, center + half_len], axis=0)
