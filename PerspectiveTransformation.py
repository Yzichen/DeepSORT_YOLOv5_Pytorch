import cv2
import numpy as np

class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self, img_size=(1120, 1080), max_dist=60):
        """Init PerspectiveTransformation."""
        self.src = np.array([(1024, 603),    # top-left
                        (1052, 603),    # top-right
                        (1415, 1080),   # bottom-right
                        (915, 1080)],   # bottom-left
                       dtype=np.float32)
        self.dst = np.array([[(510, 0),
                         (610, 0),
                         (610, 1080),
                         (510, 1080)]],
                       dtype=np.float32)   # 60m对应1080px,  1.435m对应100px
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

        self.max_dist = max_dist
        self.img_size = img_size
        self.px2meters_y = max_dist / 1080      # 60m对应1080px,  1.435m对应100px
        self.px2meters_x = 1.435 / 100

    def forward(self, img, flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
            img (np.array): A front view image
            img_size (tuple): Size of the image (width, height)
            flags : flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M, self.img_size, flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view

        Parameters:
            img (np.array): A top view image
            img_size (tuple): Size of the image (width, height)
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Front view image
        """
        return cv2.warpPerspective(img, self.M_inv, self.img_size, flags=flags)

    def transform_points(self, points, to_real=False):
        """
        Args:
            points: (N, 2)
        Returns:
            trans_points: (N, 2)
        """
        if points.ndim == 1:
            points = points[None, :]
        # (N, 3)
        points_hom = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
        # (N, 3) @ (3, 3) --> (N, 3)
        trans_points = np.dot(points_hom, self.M.T)
        trans_points = trans_points[:, 0:2] / trans_points[:, 2:3]    # (N, 2)

        if to_real:
            points_x = trans_points[:, 0]
            points_y = trans_points[:, 1]
            points_y = self.max_dist - points_y * self.px2meters_y  # 坐标系原点转换到车头.
            points_x = (points_x - self.img_size[0] / 2) * self.px2meters_x
            trans_points = np.stack((points_x, points_y), axis=1)    # (N, 2)

        return trans_points


