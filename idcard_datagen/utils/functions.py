import cv2 as cv
import numpy as np


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


def rotate_boxes(boxes: np.ndarray, angle, cx, cy, h, w):
    boxes = boxes.tolist()
    nboxes = []
    for box in boxes:
        nbox = rotate_box(box, angle, cx, cy, h, w)
        nboxes.append(nbox)
    nboxes = np.array(nboxes)
    return nboxes


def rotate_box(box, angle, cx, cy, h, w):
    """
    source https://cristianpb.github.io/blog/image-rotation-opencv#affine-transformation
    :param bb:
    :param angle:
    :param cx:
    :param cy:
    :param h:
    :param w:
    :return:
    """
    new_bb = list(box)
    for i, coord in enumerate(box):
        # opencv calculates standard transformation matrix
        M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0], coord[1], 1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M, v)
        new_bb[i] = (calculated[0], calculated[1])
    return new_bb
