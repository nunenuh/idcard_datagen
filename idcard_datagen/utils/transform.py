import random

import cv2 as cv
import numpy as np

from idcard_datagen.utils import functions as F


class RandomRotation(object):
    """
    source of idea :
    https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
    """

    def __init__(self, angle, random=True):
        self.angle = angle
        self.random = random
        if random:
            if type(self.angle) == tuple:
                assert len(self.angle) == 2, "Invalid range"
            else:
                self.angle = (-self.angle, self.angle)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if self.random:
            angle = random.uniform(*self.angle)
        else:
            angle = self.angle

        (height, width) = image.shape[:2]
        (cx, cy) = (width // 2, height // 2)

        rotated_image = F.rotate_image(image, angle)
        rotated_boxes = F.rotate_boxes(boxes, angle, cx, cy, height, width)

        assert isinstance(rotated_image, object)
        return rotated_image, rotated_boxes, angle


class RandomShear(object):
    def __init__(self, shear_factor=0.2):
        self.shear_factor = shear_factor

    def __call__(self, image, boxes):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rotate = RandomRotation(angle=12, random=False)
    image = cv.imread('../../data/idcard/base1.png', cv.IMREAD_UNCHANGED)

    h, w = image.shape[:2]
    box = tl, tr, br, bl = (0, 0), (w, 0), (w, h), (0, h)
    boxes = np.array([box])

    rot_img, rot_boxes, angle = rotate(image, boxes)
    rot_boxes = rot_boxes.astype(np.int32)

    # rot_boxes = rot_boxes.reshape((-1, 1, 2)).astype(np.int32)


    rot_img = cv.polylines(rot_img, [rot_boxes], True, (0,0,255), 4)

    print(rot_boxes)
    print(angle)

    plt.imshow(rot_img[:,:,:3]); plt.show()
    # cv.imshow('Image', rot_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

