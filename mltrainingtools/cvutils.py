
def segment_IoU(x, w, x_h, w_h):
    if x < x_h:
        return ordered_IoU(x, w, x_h, w_h)
    else:
        return ordered_IoU(x_h, w_h, x, w)


def ordered_IoU(x, w, x_h, w_h):
    if (x_h > x + w) or (x_h + w_h < x):
        return 0
    else:
        return min(x + w, x_h + w_h) - max(x, x_h)


def IoU(x, y, h, w, x_h, y_h, h_h, w_h):
    x_inter = segment_IoU(x, w, x_h, w_h)
    y_inter = segment_IoU(y, h, y_h, h_h)

    intersection = x_inter * y_inter
    union = (h * w) + (h_h * w_h) - intersection

    return intersection/union