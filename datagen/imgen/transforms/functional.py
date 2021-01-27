import numpy as np
import cv2 as cv
import random
from numpy.core.fromnumeric import resize

from skimage import data, exposure, filters


def coin_toss(p=0.5):
    pf = 1-p
    ot, wt = [True, False], [p, pf]
    res = random.choices(population=ot, weights=wt, k=1)
    return res[0]


def rotate_image(image, angle):
    """
    source https://cristianpb.github.io/blog/image-rotation-opencv#affine-transformation

    :param image:
    :param angle:
    :return:
    """

    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


def rotate_boxes(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    # calculated = calculated.reshape(-1, 4, 2)
    calculated = calculated.reshape(-1, 8)

    return calculated


def shear_image_boxes(image, mwboxes, cboxes, shear_factor):
    h, w = image.shape[:2]
    nW = w + abs(shear_factor * h)
    nH = h + abs(shear_factor * w)
    scale_factor_x = nW / w
    scale_factor_y = nH / h

    M = np.array([[1, 0, abs(shear_factor)], [0, 1, 0]])

    image = cv.warpAffine(image, M, (int(nW), h))
    image = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)

    scale_mat = [
        scale_factor_x, 1, scale_factor_x, 1,
        scale_factor_x, 1, scale_factor_x, 1
    ]
    sheared_mwboxes = mwboxes / scale_mat
    
    cboxes_list = []
    for cb in cboxes:
        cb = cb / scale_mat
        cboxes_list.append(cb)
    sheared_cboxes = cboxes_list
    
    return image, sheared_mwboxes, sheared_cboxes


def resize_image_boxes(image, mwboxes, cboxes, size, interpolation=cv.INTER_LINEAR):
        resized_image = cv.resize(image, size, interpolation=interpolation)

        oH, oW = image.shape[:2]
        rH, rW = resized_image.shape[:2]
        scf_x, scf_y = oW / rW, oH / rH
        scale_mat = [
            scf_x, scf_y, scf_x, scf_y,
            scf_x, scf_y, scf_x, scf_y
        ]
        resized_mwboxes = mwboxes / scale_mat
        
        cboxes_list = []
        for cb in cboxes:
            cb = cb / scale_mat
            cboxes_list.append(cb)
        resized_cboxes = cboxes_list
        
        return resized_image, resized_mwboxes, resized_cboxes

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

def adjust_contrast(image, level, min=1, max=3):
    clevel = (level * (max - min) / 100) + min
    contrasted_image = image_scale_abs(image, contrast_level=clevel)
    return contrasted_image
    
def adjust_brightness(image, level):
    brightned_image = image_scale_abs(image, brightness_level=level)
    return brightned_image

def image_scale_abs(image, contrast_level=1.0, brightness_level=1):
    # alpha = 1 # Contrast control (1.0-3.0)
    # beta = 0 # Brightness control (0-100)
    alpha, beta = contrast_level, brightness_level
    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def gaussian_blur(image, sigma=1.6, ksize=(5,5)):
    img_gauss = cv.GaussianBlur(image, ksize=ksize, sigmaX=sigma, sigmaY=sigma)
    return img_gauss

def median_blur(image, ksize=5):
    img_median = cv.medianBlur(image, ksize)
    return img_median

def dilation_morphology(image, shift, iterations=1):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv.dilate(image, kernel, iterations=iterations)
    return image

def opening_morphology(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return image

def closing_morphology(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return image

def sharpen(image):
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened

def emboss(image):
    kernel=np.array([[0,-1,-1],
                     [1, 0,-1],
                     [1, 1, 0]])
    image = cv.filter2D(image, -1, kernel) + 128
    return image

def hue_shifting(image, shift=16):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    shift_h = (h + shift) % 180
    shift_hsv = cv.merge([shift_h, s, v])
    shift_img = cv.cvtColor(shift_hsv, cv.COLOR_HSV2BGR)
    return shift_img

def channel_shuffle(image):
    if len(image.shape[-1])==4:
        print('Image Shape : ',len(image.shape))
        b,g,r,a = cv.split(image)
        chan = [b,g,r]
        random.shuffle(chan)
        chan.append(a)
        rand_chan_image = cv.merge(chan)
    elif len(image.shape[-1])==3:
        print('Image Shape : ',len(image.shape))
        splitted_image = cv.split(image)
        b,g,r  = splitted_image
        chan = [b,g,r]
        random.shuffle(chan)
        rand_chan_image = cv.merge(chan)
    else:
        #reject later
        raise Exception("Image Channel must be more than BGR or BGRA, grayscale is not accepted")
    
    return rand_chan_image

def to_lo_res(image, factor=0.5, interpolation=cv.INTER_LINEAR):
    (h, w) = image.shape[:2]
    hn, wn = int(h*factor), int(w*factor) 
    image = cv.resize(image, (wn, hn), interpolation=interpolation)
    image = cv.resize(image, (w, h), interpolation=interpolation)
    return image
    
    


