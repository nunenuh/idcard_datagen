import random
from typing import *

import cv2 as cv
import numpy as np

from . import functional as F
from . import effects as E
from ..ops import math_ops, image_ops, boxes_ops

from skimage.util import noise, random_noise

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
class ComposeRandomChoice(Compose):
    def __init__(self, transforms, k=1, debug=False):
        super(ComposeRandomChoice, self).__init__(transforms)
        self.k = k 
        self.debug = debug
        self.transforms_fn = random.sample(self.transforms, k=self.k)
    
    def __call__(self, img):
        self.transforms_fn = random.sample(self.transforms, k=self.k)
        if self.debug: print(self.transforms_fn)
        for t in self.transforms_fn:
            img = t(img)
        return img
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info+ f'({self.transforms_fn})'
        return info  
    
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
    


class RandomNoise(object):
    def __init__(self, amount_range=(0.07, 0.26), mode_choice=('pepper','s&p'),
                 randomize=True, p=0.5):
        self.amount_range = amount_range
        self.mode_choice = mode_choice
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        self.mode = random.choice(self.mode_choice)
        self.amount = random.uniform(*self.amount_range)
        
        if self.randomize and F.coin_toss(p=self.rand_prob):
            if image.shape[-1]==4:
                b,g,r,a = cv.split(image)
                bgr = cv.merge([b,g,r])
                
                noisy = random_noise(bgr, mode=self.mode, amount=self.amount)
                noisy = (noisy * 255).astype(np.uint8)
                b,g,r = cv.split(noisy)
                noisy = cv.merge([b,g,r,a])
            
            else:
                noisy = random_noise(image, mode=self.mode, amount=self.amount)
                noisy = (noisy * 255).astype(np.uint8)
        else:
            noisy = image
            
        return noisy
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info+ f'(amount_range={self.amount_range}, mode={self.mode_choice})'
        return info  


class Darken(object):
    def __init__(self, level=-10, gamma=0.5):
        self.level = level
        self.gamma = gamma
        
    def __call__(self, image):
        img_lo_ct = F.adjust_contrast(image, self.level)
        img_lo_br = F.adjust_brightness(img_lo_ct, self.level)
        img_lo_gm = F.adjust_gamma(img_lo_br, gamma=self.gamma)
        
        return img_lo_gm
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(level={self.level}, gamma={self.gamma})'
        return info  
    
class Lighten(object):
    def __init__(self, level=13, gamma=1.1):
        self.level = level
        self.gamma = gamma
    
    def __call__(self, image):
        img_hi_ct = F.adjust_contrast(image, self.level)
        img_hi_br = F.adjust_brightness(img_hi_ct, self.level)
        img_hi_gm = F.adjust_gamma(img_hi_br, gamma=self.gamma)
        
        return img_hi_gm
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(level={self.level}, gamma={self.gamma})'
        return info  
    
    
class ToLoRes(object):
    def __init__(self, factor=0.5):
        self.factor = factor
    
    def __call__(self, image):
        image = F.to_lo_res(image, factor=self.factor)
        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(factor={self.factor})'
        return info  
    
class RandomLoRes(object):
    def __init__(self, factor_range=(0.4,0.6), 
                 randomize=True, p=0.5):
        self.factor_range = factor_range
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        self.factor = random.uniform(*self.factor_range)
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.to_lo_res(image, factor=self.factor)
        return image

    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(factor_range={self.factor_range})'
        return info  
        

class RandomGamma(object):
    def __init__(self, gamma_range=(1.0, 2.0), 
                 randomize=True, p=0.5):
        self.gamma_range = gamma_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        self.gamma = random.uniform(*self.gamma_range)
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.adjust_gamma(image, self.gamma)

        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(gamma_range={self.gamma_range}, p={self.rand_prob})'
        return info  
    

class RandomContrast(object):
    def __init__(self, level_range=(10,30), 
                 randomize=True, p=0.5):
        self.level_range = level_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        self.level = int(random.uniform(*self.level_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.adjust_contrast(image, self.level)

        return image
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(level_range={self.level_range}, p={self.rand_prob})'
        return info    

            

class RandomBrightness(object):
    def __init__(self, level_range=(10,30), 
                 randomize=True, p=0.5):
        self.level_range = level_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        self.level = int(random.uniform(*self.level_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.adjust_brightness(image, self.level)

        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(level_range={self.level_range}, p={self.rand_prob})'
        return info    

    
class RandomHueShifting(object):
    def __init__(self, shift_range=(1, 100), 
                 randomize=True, p=0.5):
        self.shift_range = shift_range
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        self.shift = int(random.uniform(*self.shift_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.hue_shifting(image, self.shift)
        
        return image
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(shift_range={self.shift_range}, p={self.rand_prob})'
        return info    
    

class RandomChannelShuffle(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.channel_shuffle(image)

        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(p={self.rand_prob})'
        return info         
    
    
class RandomSharpen(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.sharpen(image)

        return image
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(p={self.rand_prob})'
        return info         
    
    
class RandomEmboss(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
    
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.emboss(image)

        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(p={self.rand_prob})'
        return info  
    
    
class RandomGaussionBlur(object):
    def __init__(self, sigma_range=(1.0, 5.0), ksize=(5,5),
                 randomize=True, p=0.5):
        self.sigma_range = sigma_range
        self.ksize = ksize
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        self.sigma = int(random.uniform(*self.sigma_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.gaussian_blur(image, sigma=self.sigma, ksize=self.ksize)

        return image
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(sigma_range={self.sigma_range}, ksize={self.ksize}, p={self.rand_prob})'
        return info  
    
class RandomMedianBlur(object):
    def __init__(self, ksize=5, randomize=True, p=0.5):
        self.ksize = ksize
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.median_blur(image, ksize=self.ksize)
        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(ksize={self.ksize}, p={self.rand_prob})'
        return info  
    
class RandomMorphDilation(object):
    def __init__(self, shift_range=(1,3), iterations=1,
                 randomize=True, p=0.5):
        self.shift_range = shift_range
        self.iterations = iterations
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        self.shift = int(random.uniform(*self.shift_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.dilation_morphology(image, shift=self.shift,
                                          iterations=self.iterations)
        return image
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info + f'(shift_range={self.shift_range}, iterations={self.iterations}, p={self.rand_prob})'
        return info  
    
class RandomMorphOpening(object):
    def __init__(self, shift_range=(1,7), randomize=True, p=0.5):
        self.shift_range = shift_range
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        self.shift = int(random.uniform(*self.shift_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.opening_morphology(image, shift=self.shift)
        return image
    
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info+ f'(shift={self.shift_range}, p={self.rand_prob})'
        return info

class RandomMorphClosing(object):
    def __init__(self, shift_range=(1,5), randomize=True, p=0.5):
        self.shift_range = shift_range
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        self.shift = int(random.uniform(*self.shift_range))
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.closing_morphology(image, shift=self.shift)
        return image     
        
    def __repr__(self):
        info = f'{self.__class__.__name__}'
        info = info+ f'(shift_range={self.shift_range}, p={self.rand_prob})'
        return info   
        

class RandomErase(object):
    def __init__(self, area_range=(0.3, 1.0), randomize=True, p=0.5):
        self.area_range = area_range

        self.rand_prob = p
        self.randomize = randomize
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.random_erasing(image, area_range=self.area_range)
        return image
    
    
class RandomShadow(object):
    def __init__(self, area_range=(0.3, 1.0), level=-10, gamma=0.5, 
                 randomize=True, p=0.5):
        self.area_range = area_range
        self.level = level
        self.gamma = gamma
        self.rand_prob = p
        self.randomize = randomize
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = F.random_shadow(image, area_range=self.area_range, 
                                    level=self.level, gamma=self.gamma)
        return image
        

class RandomAddSunFlares(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_sun_flare(image)
        return image 
    
    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info  

class RandomAddShadow(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_shadow(image)
        return image

    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info  
    
class RandomAddFog(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_fog(image)
        return image

    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info  

class RandomAddSnow(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_snow(image)
        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info  
    
class RandomAddRain(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_rain(image)
        return image
        
    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info  
    
class RandomAddSpeed(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            try:
                image = E.add_speed(image)
            except:
                pass
        return image  
        
    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info  

class RandomAddGravel(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_gravel(image)
        return image   
        
    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info          

class RandomAddAutumn(object):
    def __init__(self, randomize=True, p=0.5):
        self.randomize = randomize
        self.rand_prob = p
        
    def __call__(self, image):
        if self.randomize and F.coin_toss(p=self.rand_prob):
            image = E.add_autumn(image)
        return image  
        
    def __repr__(self):
        info = f'{self.__class__.__name__}(p={self.rand_prob})'
        return info           


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
        
        
    def __repr__(self):
        info = f'{self.__class__.__name__}(size={self.size})'
        return info  


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

    def __call__(self, image: np.ndarray, mwboxes: np.ndarray, cboxes:List[np.ndarray] ) -> Tuple[Any, Any]:
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
        rotated_mwboxes = F.rotate_boxes(mwboxes, angle, cx, cy, h, w)
        
        resized_image = cv.resize(rotated_image, (w, h), interpolation=cv.INTER_LINEAR)

        scale_factor_x = rotated_image.shape[1] / w
        scale_factor_y = rotated_image.shape[0] / h
        scaled_mat = [
            scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y,
            scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y,
        ]
        resized_mwboxes = rotated_mwboxes / scaled_mat
        
        cboxes_list = []
        for cb in cboxes:
            if len(cb)>0:
                cb = F.rotate_boxes(cb, angle, cx, cy, h, w)
                cb = cb / scaled_mat
                cboxes_list.append(cb)
        resized_cboxes = cboxes_list

        return resized_image, resized_mwboxes, resized_cboxes, angle


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

    def __call__(self, image, mwboxes, cboxes):
        if self.randomize:
            if F.coin_toss(p=self.rand_prob):
                shear_factor = random.uniform(*self.shear_factor)
            else:
                shear_factor = 0.0
        else:
            shear_factor = self.shear_factor
            
        image, mwboxes, cboxes = F.shear_image_boxes(image, mwboxes, cboxes, shear_factor)

        return image, mwboxes, cboxes, shear_factor


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


    def __call__(self, image, mwboxes, cboxes):
        rimage, rmwboxes, rcboxes, angle = self.rotate_fn(image, mwboxes, cboxes)
        simage, smwboxes, scboxes, factor = self.shear_fn(rimage, rmwboxes, rcboxes)

        self.rotation_angle = angle
        self.shear_factor = factor

        return simage, smwboxes, scboxes


light_effect_fn = ComposeRandomChoice([
    RandomGamma(gamma_range=(0.5, 2.5), p=0.5),
    RandomContrast(level_range=(0, 20), p=0.5),
    RandomBrightness(level_range=(20, 50), p=0.5),
], k=2, debug=False)

darklight_effect_fn = ComposeRandomChoice([
    Darken(),
    Lighten()
], k=1, debug=False)

color_effect_fn = ComposeRandomChoice([
    RandomHueShifting(shift_range=(1, 100), p=0.5),
    RandomChannelShuffle(p=0.5),
], k=1, debug=False)

morph_effect_fn = ComposeRandomChoice([
    RandomMorphDilation(p=0.5),
    RandomMorphOpening(p=0.5),
    RandomMorphClosing(p=0.5)
], k=1, debug=False)

simple_basic_effect_fn = ComposeRandomChoice([
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=0.5),
    RandomSharpen(p=0.5),
    RandomNoise(amount_range=(0.05, 0.06), p=0.5),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=0.5),
], k=1, debug=False)


medium_basic_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06)),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=3, debug=False)


complex_basic_effect_fn = ComposeRandomChoice([
    light_effect_fn,
    darklight_effect_fn,
    color_effect_fn,
    RandomLoRes(factor_range=(0.3, 0.5), p=1),
    RandomSharpen(p=1),
    RandomNoise(amount_range=(0.05, 0.06), p=1),
    RandomGaussionBlur(sigma_range=(1.0, 5.0), p=1),
    morph_effect_fn,
], k=5, debug=False)

basic_effect_dict = {
    'simple': simple_basic_effect_fn, 
    "medium": medium_basic_effect_fn,
    "complex": complex_basic_effect_fn,
}

simple_advance_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(amount_range=(0.05, 0.07), p=0.5),
], k=1, debug=False)

medium_advance_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomAddShadow(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(p=0.5),
    RandomAddSnow(p=0.5),
    RandomAddRain(p=0.5),
    RandomAddSpeed(p=0.5),
    RandomAddFog(p=0.5),
    RandomAddGravel(p=0.5),
], k=3, debug=False)

complex_advance_effect_fn = ComposeRandomChoice([
    RandomAddSunFlares(p=0.5),
    RandomAddShadow(p=0.5),
    RandomShadow(p=0.5),
    RandomNoise(p=0.5),
    RandomAddSnow(p=0.5),
    RandomAddRain(p=0.5),
    RandomAddSpeed(p=0.5),
    RandomAddFog(p=0.5),
    RandomAddGravel(p=0.5),
], k=5, debug=False)

advance_effect_dict = {
    "simple": simple_advance_effect_fn,
    "medium": medium_advance_effect_fn,
    "complex": complex_advance_effect_fn,
}

class AugmentGenerator(object):
    def __init__(self, scale_ratio: float = 0.25, angle: int = 45, shear_factor: float = 0.3,
                 use_basic_effect: bool = True, basic_effect_mode: str = "simple", 
                 use_adv_effect: bool = True, adv_effect_mode: str = "simple",
                 randomize: bool = True, rand_prob: float = 0.5):
        self.scale_ratio = scale_ratio
        self.angle = angle
        self.shear_factor = shear_factor
        self.randomize = randomize
        self.rand_prob = rand_prob
        self.use_basic_effect = use_basic_effect
        self.basic_effect_mode = basic_effect_mode
        self.use_adv_effect = use_adv_effect
        self.adv_effect_mode = adv_effect_mode
        
        self._init_objects_call_fn()
        self._init_effect_fn()
        
    
    def _init_effect_fn(self):
        basic_effect_mode = self.basic_effect_mode
        self.basic_effect_fn = basic_effect_dict[basic_effect_mode]
        
        adv_effect_mode = self.adv_effect_mode
        self.advance_effect_fn = advance_effect_dict[adv_effect_mode]
        
    
    def _init_objects_call_fn(self):
        self.random_rotate_shear_fn = RandomRotateAndShear(
            self.angle, self.shear_factor, 
            randomize=self.randomize, 
            rand_prob=self.rand_prob
        )

    def __call__(self, background_image, foreground_image, mwboxes: np.ndarray = None, cboxes:List[np.ndarray] = None):
        # boxes = F.corner_from_shape(foreground_image)
       
        foreground_image, mwboxes, cboxes = self.random_rotate_shear_fn(foreground_image, mwboxes, cboxes)
        self.actual_angle = self.random_rotate_shear_fn.rotation_angle
        self.actual_shear = self.random_rotate_shear_fn.shear_factor

        bgH, bgW = back_size = background_image.shape[:2]
        fgH, fgW = frgd_size = foreground_image.shape[:2]
        nfgdH, nfgdW = nfgd_size = math_ops.scale_size_ratio(back_size, frgd_size, ratio=self.scale_ratio)
        xmin, ymin, xmax, ymax = xybox = math_ops.random_safe_box_location(back_size, nfgd_size)

        # frgd_base_image = cv.resize(foreground_image, dsize=(nfgdW, nfgdH), interpolation=cv.INTER_LINEAR)
        # resize_image_boxes = ResizeImageBoxes(size=(nfgdW, nfgdH))
        
        frgd_base_image, mwboxes, cboxes = F.resize_image_boxes(foreground_image, mwboxes, cboxes,  size=(nfgdW, nfgdH))
        frgd_segment_image = cv.split(frgd_base_image)[-1]
        frgd_segment_image = (frgd_segment_image > 0).astype(np.uint8)
        
        segment_canvas = image_ops.create_canvas(back_size,  dtype=np.uint8)
        segment_image = image_ops.join2image_withcoords(frgd_segment_image, segment_canvas, xybox)

        if self.use_basic_effect:
            frgd_base_image = self.basic_effect_fn(frgd_base_image)
            
        
        if frgd_base_image.shape[-1] != 4:
            frgd_base_image = cv.cvtColor(frgd_base_image, cv.COLOR_BGR2BGRA)
            
        overlay_canvas = image_ops.create_canvas((bgH, bgW, 4), dtype=np.uint8)
        overlay_image = image_ops.join2image_withcoords(frgd_base_image, overlay_canvas, xybox)
        
        composite_image = image_ops.composite2image(background_image, overlay_image)
        composite_image = composite_image.astype(np.uint8)
        
        # print('Composite Image shape before basic effect',composite_image.shape)
        
        
        if self.use_adv_effect:
            if self.use_basic_effect:
                composite_image = self.basic_effect_fn(composite_image)
            # print('Composite Image shape after basic effect',composite_image.shape)
            composite_image = self.advance_effect_fn(composite_image) 
            
            

        mwboxes = boxes_ops.boxes_reorder(mwboxes)
        mwboxes = mwboxes + [xmin, ymin]
        
        cboxes_list = []
        for cb in cboxes:
            cb = boxes_ops.boxes_reorder(cb)
            cb = cb + [xmin, ymin]
            cboxes_list.append(cb)
        cboxes = cboxes_list
        
        

        return segment_image, composite_image, mwboxes, cboxes


if __name__ == "__main__":
    # rotate = RandomRotation(angle=12, randomize=True)
    # shear = RandomShear(shear_factor=0.9, randomize=True)
    # # resize = Resize(size=(500, 700))
    # random_augment = RandomAugment(angle=35, shear_factor=0.5)

    # image = cv.imread('../../data/idcard/base1.png', cv.IMREAD_UNCHANGED)

    # h, w = image.shape[:2]

    # box = [0, 0, w, h]
    # boxes = np.array([box])
    # boxes = F.get_corners(boxes)

    # # rimage, rboxes = rotate(image, boxes)
    # # simage, sboxes = shear(rimage, rboxes)
    # # rz_image, rz_boxes = resize(simage, sboxes)

    # # print(sboxes.astype(np.int32))
    # nimage, nboxes = random_augment(image, boxes)

    # # nimage = rz_image
    # nboxes = F.boxes_reorder(nboxes)
    # # print(nboxes)

    # polyline_boxes = nboxes.reshape((-1, 1, 2)).astype(np.int32)
    # nimage = cv.polylines(nimage, [polyline_boxes], True, (0, 0, 255), 4)

    # import matplotlib.pyplot as plt

    # plt.imshow(nimage[:, :, :3]);
    # plt.show()
    # print(f'Boxes: \n{nboxes}')
    # print(f'Image Shape: {nimage.shape}')

    pass