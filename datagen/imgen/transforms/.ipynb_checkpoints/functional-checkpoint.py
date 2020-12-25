import numpy as np
import cv2 as cv
import random


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


def shear_image_boxes(image, boxes, shear_factor):
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
    boxes = boxes / scale_mat
    
    return image, boxes



