import cv2 as cv
import numpy as np
from ...config import font_config

__all__ = ['generate_image', 'join2image_withcoords', 'create_overlay_image',
           'create_segmentation_image', 'composite2image',
           'font_path','create_canvas', 'image_selection']


def font_path(name="arial"):
    font_config.font_path.get(name)


def generate_image(size, fcall=np.zeros, dtype=None):
    out = fcall(size)
    if dtype: out = fcall(size, dtype)
    return out


def image_selection(img: np.ndarray, val: int = 0):
    """
    make selection from image with specified value
    :param img: numpy image
    :param val: integer number from 0 to 255
    :return: grayscale image
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return (gray > val).astype(np.uint8)


def create_canvas(size, fcall=np.zeros, dtype=None):
    if dtype:
        canvas = fcall(size).astype(dtype)
    else:
        canvas = fcall(size)
    return canvas


def join2image_withcoords(src_img, dst_img, xybox):
    out = dst_img.copy()
    xmin, ymin, xmax, ymax = xybox
    out[ymin:ymax, xmin:xmax] = src_img
    return out


def create_overlay_image(image, dst_size, bsn_dsize, xybox):
    dst_image = generate_image((dst_size[0], dst_size[1], 4), dtype=np.uint8)
    src_image = cv.resize(image.copy(), dsize=bsn_dsize, interpolation=cv.INTER_CUBIC)
    return join2image_withcoords(src_image, dst_image, xybox)


def create_segmentation_image(image, dst_size, bsn_dsize, xybox):
    dst_image = generate_image(dst_size)
    src_image = cv.resize(image.copy(), dsize=bsn_dsize, interpolation=cv.INTER_CUBIC)
    return join2image_withcoords(src_image, dst_image, xybox)


def composite2image(background_image, overlay_image):
    """
    composite two image with jpg and png extension with same size.

    :param background_image: rgb image (jpeg image), numpy (h,w,3)
    :param overlay_image: rgba image (png image), numpy (h,w,4)
    :return: numpy (h,w,3) image
    """
    img1, img2 = background_image.copy(), overlay_image.copy()
    h, w = img1.shape[:2]
    result = np.zeros((h, w, 3), np.uint8)

    alpha = img2[:, :, 3] / 255.0
    result[:, :, 0] = (1. - alpha) * img1[:, :, 0] + alpha * img2[:, :, 0]
    result[:, :, 1] = (1. - alpha) * img1[:, :, 1] + alpha * img2[:, :, 1]
    result[:, :, 2] = (1. - alpha) * img1[:, :, 2] + alpha * img2[:, :, 2]

    return result
