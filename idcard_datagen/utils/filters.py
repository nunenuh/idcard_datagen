import cv2 as cv
import numpy as np

__all__ = ['image_selection']


def image_selection(img: np.ndarray, val: int = 0):
    """
    make selection from image with specified value
    :param img: numpy image
    :param val: integer number from 0 to 255
    :return: grayscale image
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return (gray > val).astype(np.uint8)
