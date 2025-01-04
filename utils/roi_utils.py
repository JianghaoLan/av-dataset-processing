

def get_offset(roi, x_ratio=0., y_ratio=0.):
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    offset_x = w * x_ratio
    offset_y = h * y_ratio
    new_roi = roi.copy()
    new_roi[[0, 2]] += offset_x
    new_roi[[1, 3]] += offset_y
    return new_roi
