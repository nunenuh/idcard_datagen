import random
from typing import *

import cv2 as cv
import numpy as np

from . import functional as F
from ..ops import math_ops, image_ops, boxes_ops


class RandomResize(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=cv.INTER_LINEAR):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        pass


class ResizeImageBoxes(object):
    def __init__(self, size: Tuple[int, int], interpolation=cv.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, boxes):
        resized_image = cv.resize(image, self.size, interpolation=self.interpolation)

        oH, oW = image.shape[:2]
        rH, rW = resized_image.shape[:2]
        scf_x, scf_y = oW / rW, oH / rH
        scale_mat = [
            scf_x, scf_y, scf_x, scf_y,
            scf_x, scf_y, scf_x, scf_y
        ]
        resized_boxes = boxes / scale_mat

        return resized_image, resized_boxes


class RandomRotation(object):
    """
    source of idea :
    https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
    """

    def __init__(self, angle, randomize=True, rand_prob=0.5):
        self.angle = angle
        self.randomize = randomize
        self.rand_prob = rand_prob
        if randomize:
            if type(self.angle) == tuple:
                assert len(self.angle) == 2, "Invalid range"
            else:
                self.angle = (-self.angle, self.angle)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[Any, Any]:
        if self.randomize:
            if F.coin_toss(p=self.rand_prob):
                angle = random.uniform(*self.angle)
            else:
                angle = 0
        else:
            angle = self.angle

        w, h = image.shape[1], image.shape[0]
        cx, cy = w // 2, h // 2

        rotated_image = F.rotate_image(image, angle)
        rotated_boxes = F.rotate_boxes(boxes, angle, cx, cy, h, w)
        resized_image = cv.resize(rotated_image, (w, h), interpolation=cv.INTER_LINEAR)

        scale_factor_x = rotated_image.shape[1] / w
        scale_factor_y = rotated_image.shape[0] / h
        scaled_mat = [
            scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y,
            scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y,
        ]
        resized_boxes = rotated_boxes / scaled_mat

        return resized_image, resized_boxes, angle


class RandomShear(object):
    def __init__(self, shear_factor=0.2, randomize=True, rand_prob=0.5):
        self.shear_factor = shear_factor
        self.randomize = randomize
        self.rand_prob = rand_prob
        if randomize:
            if type(self.shear_factor) == tuple:
                assert len(self.shear_factor) == 2, "Invalid range for scaling factor"
            else:
                self.shear_factor = (-self.shear_factor, self.shear_factor)

    def __call__(self, image, boxes):
        if self.randomize:
            if F.coin_toss(p=self.rand_prob):
                shear_factor = random.uniform(*self.shear_factor)
            else:
                shear_factor = 0.0
        else:
            shear_factor = self.shear_factor
            
        image, boxes = F.shear_image_boxes(image, boxes, shear_factor)

        return image, boxes, shear_factor


class RandomAugment(object):
    def __init__(self, angle=45, shear_factor=0.3, 
                 randomize=True, rand_prob=0.5):
        self.angle = angle
        self.shear_factor = shear_factor
        self.randomize = randomize
        self.rand_prob = rand_prob

    def __call__(self, image, boxes):
        rotate = RandomRotation(self.angle, randomize=self.randomize, rand_prob=self.rand_prob)
        shear = RandomShear(shear_factor=self.shear_factor, randomize=self.randomize, rand_prob=self.rand_prob)

        rimage, rboxes, angle = rotate(image, boxes)
        simage, sboxes, factor = shear(rimage, rboxes)

        self.rotation_angle = angle
        self.shear_factor = factor

        return simage, sboxes


class AugmentGenerator(object):
    def __init__(self, scale_ratio=0.25, angle=45, shear_factor=0.3, 
                 randomize=True, rand_prob=0.5):
        self.scale_ratio = scale_ratio
        self.angle = angle
        self.shear_factor = shear_factor
        self.randomize = randomize
        self.rand_prob = rand_prob

    def __call__(self, background_image, foreground_image, boxes: np.ndarray = None):
        # boxes = F.corner_from_shape(foreground_image)
        random_augment = RandomAugment(
            self.angle, self.shear_factor, 
            randomize=self.randomize, 
            rand_prob=self.rand_prob
        )
        foreground_image, boxes = random_augment(foreground_image, boxes)

        self.actual_angle = random_augment.rotation_angle
        self.actual_shear = random_augment.shear_factor


        bgH, bgW = back_size = background_image.shape[:2]
        fgH, fgW = frgd_size = foreground_image.shape[:2]
        nfgdH, nfgdW = nfgd_size = math_ops.scale_size_ratio(back_size, frgd_size, ratio=self.scale_ratio)
        xmin, ymin, xmax, ymax = xybox = math_ops.random_safe_box_location(back_size, nfgd_size)

        # frgd_base_image = cv.resize(foreground_image, dsize=(nfgdW, nfgdH), interpolation=cv.INTER_LINEAR)
        resize_image_boxes = ResizeImageBoxes(size=(nfgdW, nfgdH))
        frgd_base_image, boxes = resize_image_boxes(foreground_image, boxes)
        frgd_segment_image = image_ops.image_selection(frgd_base_image, val=0)

        segment_canvas = image_ops.create_canvas(back_size)
        segment_image = image_ops.join2image_withcoords(frgd_segment_image, segment_canvas, xybox)

        overlay_canvas = image_ops.create_canvas((bgH, bgW, 4), dtype=np.uint8)
        overlay_image = image_ops.join2image_withcoords(frgd_base_image, overlay_canvas, xybox)
        composite_image = image_ops.composite2image(background_image, overlay_image)

        boxes = boxes_ops.boxes_reorder(boxes)
        boxes = boxes + [xmin, ymin]

        return segment_image, composite_image, boxes,


if __name__ == "__main__":
    rotate = RandomRotation(angle=12, randomize=True)
    shear = RandomShear(shear_factor=0.9, randomize=True)
    # resize = Resize(size=(500, 700))
    random_augment = RandomAugment(angle=35, shear_factor=0.5)

    image = cv.imread('../../data/idcard/base1.png', cv.IMREAD_UNCHANGED)

    h, w = image.shape[:2]

    box = [0, 0, w, h]
    boxes = np.array([box])
    boxes = F.get_corners(boxes)

    # rimage, rboxes = rotate(image, boxes)
    # simage, sboxes = shear(rimage, rboxes)
    # rz_image, rz_boxes = resize(simage, sboxes)

    # print(sboxes.astype(np.int32))
    nimage, nboxes = random_augment(image, boxes)

    # nimage = rz_image
    nboxes = F.boxes_reorder(nboxes)
    # print(nboxes)

    polyline_boxes = nboxes.reshape((-1, 1, 2)).astype(np.int32)
    nimage = cv.polylines(nimage, [polyline_boxes], True, (0, 0, 255), 4)

    import matplotlib.pyplot as plt

    plt.imshow(nimage[:, :, :3]);
    plt.show()
    print(f'Boxes: \n{nboxes}')
    print(f'Image Shape: {nimage.shape}')
