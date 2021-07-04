
from typing import *

import cv2 as cv
import numpy as np

from ..ops import math_ops, image_ops, boxes_ops
from . import functional as F
from . import effects as E
from . import tpack
from .transforms import *


class AugmentGenerator(object):
    def __init__(self, scale_ratio: float = 0.25, angle: int = 45, shear_factor: float = 0.3,
                 foreground_fx: str = "simple",  background_fx: str = None, 
                 composite_bfx: str = None, composite_afx: str = "simple",
                 randomize: bool = True, rand_prob: float = 0.5):
        self.scale_ratio = scale_ratio
        self.angle = angle
        self.shear_factor = shear_factor
        self.randomize = randomize
        self.rand_prob = rand_prob
        
        self.foreground_fx = foreground_fx
        self.background_fx = background_fx
        self.composite_bfx = composite_bfx
        self.composite_afx = composite_afx
        
        self._init_objects_call_fn()
        self._init_effect_fn()
        
    def _init_effect_fn(self):
        self.foreground_fx = tpack.foreground_effect_dict.get(self.foreground_fx, None)
        self.background_fx = tpack.background_effect_dict.get(self.background_fx, None)
        self.composite_bfx = tpack.composite_bfx_effect_dict.get(self.composite_bfx, None)
        self.composite_afx = tpack.composite_afx_effect_dict.get(self.composite_afx, None)
        
        self.info_fx = {
            "foreground_fx": [],
            "background_fx": [],
            "composite_bfx": [],
            "composite_afx": [],
        }
        
    
    def _init_objects_call_fn(self):
        self.random_rotate_shear_fn = RandomRotateAndShear(
            self.angle, self.shear_factor, 
            randomize=self.randomize, 
            rand_prob=self.rand_prob
        )
        
    def _create_segment_image(self, frgd_base_image, backgrd_size, xybox):
        frgd_segment_image = cv.split(frgd_base_image)[-1]
        frgd_segment_image = (frgd_segment_image > 0).astype(np.uint8)
        segment_canvas = image_ops.create_canvas(backgrd_size,  dtype=np.uint8)
        segment_image = image_ops.join2image_withcoords(frgd_segment_image, segment_canvas, xybox)
        
        return segment_image
    
    def _create_composite_image(self, background_image, frgd_base_image, xybox):
        (bgH, bgW) = background_image.shape[:2]
        overlay_canvas = image_ops.create_canvas((bgH, bgW, 4), dtype=np.uint8)
        overlay_image = image_ops.join2image_withcoords(frgd_base_image, overlay_canvas, xybox)
        
        composite_image = image_ops.composite2image(background_image, overlay_image)
        composite_image = composite_image.astype(np.uint8)
        
        return composite_image

    def _reorder_boxes(self, mwboxes, cboxes, xybox):
        xmin, ymin, xmax, ymax = xybox
        mwboxes = boxes_ops.boxes_reorder(mwboxes)
        mwboxes = mwboxes + [xmin, ymin]
        
        cboxes_list = []
        for cb in cboxes:
            cb = boxes_ops.boxes_reorder(cb)
            cb = cb + [xmin, ymin]
            cboxes_list.append(cb)
        cboxes = cboxes_list
        
        return mwboxes, cboxes
    
    def _nforegrd_size_with_xybox(self, back_size, frgd_size):
        nfgd_size = math_ops.scale_size_ratio(back_size, frgd_size, ratio=self.scale_ratio)
        xybox = math_ops.random_safe_box_location(back_size, nfgd_size)
        return nfgd_size, xybox
        

    def __call__(self, background_image, foreground_image, mwboxes: np.ndarray = None, cboxes:List[np.ndarray] = None):
        backgrd_image = background_image.copy()
        foregrd_image = foreground_image.copy()
        
        foregrd_image, mwboxes, cboxes = self.random_rotate_shear_fn(foregrd_image, mwboxes, cboxes)
        self.actual_angle = self.random_rotate_shear_fn.rotation_angle
        self.actual_shear = self.random_rotate_shear_fn.shear_factor
        self.info_fx['foreground_fx'].append(self.random_rotate_shear_fn.info)

        backgrd_size, foregrd_size = backgrd_image.shape[:2], foregrd_image.shape[:2]
        
        nfgd_size, xybox = self._nforegrd_size_with_xybox(backgrd_size, foregrd_size)
        nfgd_height, nfgd_weight = nfgd_size
        
        foregrd_image, mwboxes, cboxes = F.resize_image_boxes(foregrd_image, mwboxes, cboxes,  size=(nfgd_weight, nfgd_height))
        segment_image = self._create_segment_image(foregrd_image, backgrd_size, xybox) 

        #safe place to use effect, becaause its after segmetntation
        if self.background_fx:
            backgrd_image = self.background_fx(backgrd_image)
            self.info_fx['background_fx'] = self.background_fx.info
            
        if self.foreground_fx:
            foregrd_image = self.foreground_fx(foregrd_image)
            self.info_fx['foreground_fx'].extend(self.foreground_fx.info)
            
        
        if foregrd_image.shape[-1] != 4:
            foregrd_image = cv.cvtColor(foregrd_image, cv.COLOR_BGR2BGRA)
        
        # preparing composite image
        composite_image = self._create_composite_image(backgrd_image, foregrd_image, xybox)
        
        if self.composite_bfx:
            composite_image = self.composite_bfx(composite_image)
            self.info_fx['composite_bfx'] = self.composite_bfx.info
            
        
        if self.composite_afx:
            composite_image = self.composite_afx(composite_image)
            self.info_fx['composite_afx'] = self.composite_afx.info
            
        
        
        mwboxes, cboxes = self._reorder_boxes(mwboxes, cboxes, xybox)
        
        return segment_image, composite_image, mwboxes, cboxes
