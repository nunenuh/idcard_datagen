import random
from typing import *

import cv2 as cv
import numpy as np

from . import functional as F
from . import effects as E
from ..ops import math_ops, image_ops, boxes_ops


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
class ComposeRandomChoice(Compose):
    def __init__(self, transforms, k=1):
        super(ComposeRandomChoice, self).__init__(transforms)
        self.k = k 
    
    def __call__(self, img):
        self.transforms = random.sample(self.transforms, k=self.k)
        for t in self.transforms:
                img = t(img)
        return img
    
class ComposeMulti(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes
    
class ComposeMultiRandomChoice(ComposeMulti):
    def __init__(self, transforms, k=1):
        super(ComposeRandomChoice, self).__init__(transforms)
        self.k = k 
    
    def __call__(self, img, boxes):
        self.transforms = random.sample(self.transforms, k=self.k)
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes
    
class RandomGamma(object):
    def __init__(self, gamma_range=(1.0, 2.0), 
                 randomize=True, p=0.5):
        self.gamma = gamma_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.gamma = random.uniform(*self.gamma)
            image = F.adjust_gamma(image, self.gamma)

        return image
    

class RandomContrast(object):
    def __init__(self, level_range=(10,30), 
                 randomize=True, p=0.5):
        self.level = level_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.level = int(random.uniform(*self.level))
            image = F.adjust_contrast(image, self.level)

        return image
            

class RandomBrightness(object):
    def __init__(self, level_range=(10,30), 
                 randomize=True, p=0.5):
        self.level = level_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.level = int(random.uniform(*self.level))
            image = F.adjust_brightness(image, self.level)

        return image
    
class RandomHueShifting(object):
    def __init__(self, shift_range=(1,100), 
                 randomize=True, p=0.5):
        self.shift = shift_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.shift = int(random.uniform(*self.shift))
            image = F.hue_shifting(image, self.shift)

        return image
    

class RandomChannelShuffle(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.channel_shuffle(image)

        return image
    
class RandomSharpen(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.sharpen(image)

        return image
    
class RandomEmboss(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.emboss(image)

        return image
    
    
class RandomGaussionBlur(object):
    def __init__(self, sigma_range=(1.0, 5.0), ksize=(5,5),
                 randomize=True, p=0.5):
        self.sigma = sigma_range
        self.ksize = ksize
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.sigma = int(random.uniform(*self.sigma))
            image = F.gaussian_blur(image, sigma=self.sigma, ksize=self.ksize)

        return image
    
class RandomMedianBlur(object):
    def __init__(self, ksize=5, randomize=True, p=0.5):
        self.ksize = ksize
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.median_blur(image, ksize=self.ksize)
        return image
    
class RandomMorphDilation(object):
    def __init__(self, shift_range=(1,3), iterations=1,
                 randomize=True, p=0.5):
        self.shift = shift_range
        self.iterations = iterations
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.shift = int(random.uniform(*self.shift))
            image = F.dilation_morphology(image, shift=self.shift,
                                          iterations=self.iterations)
        return image
    
class RandomMorphOpening(object):
    def __init__(self, shift_range=(1,7), randomize=True, p=0.5):
        self.shift = shift_range
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.shift = int(random.uniform(*self.shift))
            image = F.opening_morphology(image, shift=self.shift)
        return image
    

class RandomMorphClosing(object):
    def __init__(self, shift_range=(1,5), randomize=True, p=0.5):
        self.shift = shift_range
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            self.shift = int(random.uniform(*self.shift))
            image = F.closing_morphology(image, shift=self.shift)
        return image        
        

class RandomAddSunFlares(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_sun_flare(image)
        return image 

class RandomAddShadow(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_shadow(image)
        return image
    
class RandomAddFog(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_fog(image)
        return image

class RandomAddSnow(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_snow(image)
        return image
    
class RandomAddRain(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_rain(image)
        return image
    
class RandomAddSpeed(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_speed(image)
        return image  

class RandomAddGravel(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_gravel(image)
        return image           

class RandomAddAutumn(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_autumn(image)
        return image           
    

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
        rimage, rboxes = F.resize_image_boxes(
            image, 
            boxes, 
            self.size, 
            self.interpolation
        )
        return rimage, rboxes


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


class RandomRotateAndShear(object):
    def __init__(self, angle=45, shear_factor=0.3, 
                 randomize=True, rand_prob=0.5):
        self.angle = angle
        self.shear_factor = shear_factor
        self.randomize = randomize
        self.rand_prob = rand_prob
        
        self._init_objects_call_fn()
        
    def _init_objects_call_fn(self):
        self.rotate_fn = RandomRotation(self.angle, 
                                        randomize=self.randomize, 
                                        rand_prob=self.rand_prob)
        
        self.shear_fn = RandomShear(shear_factor=self.shear_factor, 
                                    randomize=self.randomize, 
                                    rand_prob=self.rand_prob)


    def __call__(self, image, boxes):
        rimage, rboxes, angle = self.rotate_fn(image, boxes)
        simage, sboxes, factor = self.shear_fn(rimage, rboxes)

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
        
        self._init_objects_call_fn()
        
    def _init_objects_call_fn(self):
        self.random_rotate_shear_fn = RandomRotateAndShear(
            self.angle, self.shear_factor, 
            randomize=self.randomize, 
            rand_prob=self.rand_prob
        )
        
        
        self.basic_effect_fn = ComposeRandomChoice([
            RandomGamma(gamma_range=(0.5, 2.5), p=0.5),
            RandomContrast(level_range=(0, 20), p=0.5),
            RandomBrightness(level_range=(20, 50), p=0.5),
            # RandomHueShifting(shift_range=(1, 100), p=0.5),
            # RandomChannelShuffle(p=0.5),
            RandomSharpen(p=0.5),
            RandomGaussionBlur(sigma_range=(1.0, 5.0), p=0.5),
            RandomMedianBlur(p=0.5),
            RandomMorphDilation(p=0.5),
            # RandomMorphOpening(p=0.5),
            # RandomMorphClosing(p=0.5)
        ], k=5)
        
        self.advance_effect_fn = ComposeRandomChoice([
            RandomAddSunFlares(p=0.5),
            RandomAddShadow(p=0.5),
            RandomAddSnow(p=0.5),
        #     RandomAddRain(p=0.5),
            RandomAddSpeed(p=0.5),
            RandomAddFog(p=0.5),
            RandomAddGravel(p=0.5),
        ], k=1)


    def __call__(self, background_image, foreground_image, boxes: np.ndarray = None):
        # boxes = F.corner_from_shape(foreground_image)
       
        foreground_image, boxes = self.random_rotate_shear_fn(foreground_image, boxes)
        self.actual_angle = self.random_rotate_shear_fn.rotation_angle
        self.actual_shear = self.random_rotate_shear_fn.shear_factor

        bgH, bgW = back_size = background_image.shape[:2]
        fgH, fgW = frgd_size = foreground_image.shape[:2]
        nfgdH, nfgdW = nfgd_size = math_ops.scale_size_ratio(back_size, frgd_size, ratio=self.scale_ratio)
        xmin, ymin, xmax, ymax = xybox = math_ops.random_safe_box_location(back_size, nfgd_size)

        # frgd_base_image = cv.resize(foreground_image, dsize=(nfgdW, nfgdH), interpolation=cv.INTER_LINEAR)
        # resize_image_boxes = ResizeImageBoxes(size=(nfgdW, nfgdH))
        
        frgd_base_image, boxes = F.resize_image_boxes(foreground_image, boxes, size=(nfgdW, nfgdH))
        frgd_segment_image = cv.split(frgd_base_image)[-1]
        frgd_segment_image = (frgd_segment_image > 0).astype(np.uint8)
        
        segment_canvas = image_ops.create_canvas(back_size,  dtype=np.uint8)
        segment_image = image_ops.join2image_withcoords(frgd_segment_image, segment_canvas, xybox)

        frgd_base_image = self.basic_effect_fn(frgd_base_image)
        overlay_canvas = image_ops.create_canvas((bgH, bgW, 4), dtype=np.uint8)
        overlay_image = image_ops.join2image_withcoords(frgd_base_image, overlay_canvas, xybox)
        
        composite_image = image_ops.composite2image(background_image, overlay_image)
        composite_image = composite_image.astype(np.uint8)
        composite_image = self.advance_effect_fn(composite_image) 

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
