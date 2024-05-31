import numpy as np


def pad_crop(image, roi):
    roi = roi.astype(int)
    x1, y1, x2, y2 = roi
    h, w = image.shape[:2]
    pad_l = max(-x1, 0)
    pad_r = max(x2 - w, 0)
    pad_t = max(-y1, 0)
    pad_b = max(y2 - h, 0)
    image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant')
    x1 = x1 + pad_l
    x2 = x2 + pad_l
    y1 = y1 + pad_t
    y2 = y2 + pad_t

    return image[y1:y2, x1:x2]
