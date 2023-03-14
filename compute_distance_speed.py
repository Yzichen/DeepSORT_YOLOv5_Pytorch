import numpy as np
import cv2
from PerspectiveTransformation import PerspectiveTransformation


transfer = PerspectiveTransformation()

def compute_distance(bottom_center):
    """
    params:
        bottom_center: (2, )
    """
    real_points = transfer.transform_points(bottom_center, to_real=True).squeeze(axis=0)   # (2, )
    dis = np.sqrt(real_points[0]**2 + real_points[1]**2)
    return dis


def compute_speed(traces, fps=30, interval=5):
    """
    params:
        traces: (N, 2)
    returns:
        vx, vy, v
    """
    if len(traces) >= (interval+1):
        p1 = traces[-(interval+1), :]    # (2, )
    else:
        p1 = traces[0, :]     # (2, )
        interval = len(traces) - 1
    p2 = traces[-1, :]

    real_p1 = transfer.transform_points(p1, to_real=True).squeeze(axis=0)   # (2, )   2: x, y
    real_p2 = transfer.transform_points(p2, to_real=True).squeeze(axis=0)   # (2, )   2: x, y
    dis_x = real_p2[0] - real_p1[0]
    dis_y = real_p2[1] - real_p1[1]

    vx = dis_x / (interval * 1 / fps) * 3.6
    vy = dis_y / (interval * 1 / fps) * 3.6
    v = np.sqrt(vx ** 2 + vy ** 2)

    return vx, vy, v

