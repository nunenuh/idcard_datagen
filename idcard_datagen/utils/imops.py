import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ['generate_image', 'join2image_withcoords', 'create_overlay_image',
           'create_segmentation_image', 'composite2image',
           'draw_text', 'draw_text_center', 'font_path',
           'get_text_position']

arial_ttf_path = '../data/fonts/arial.ttf'
ocra_ttf_path = '../data/fonts/ocr_a_ext.ttf'


def font_path(name="arial"):
    if name == "arial":
        return arial_ttf_path
    elif name == "ocra":
        return ocra_ttf_path
    else:
        return arial_ttf_path


def generate_image(size, fcall=np.zeros, dtype=None):
    out = fcall(size)
    if dtype: out = fcall(size, dtype)
    return out


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


def draw_text(np_img: np.ndarray, text: str, xymin: tuple, font_name: str = "ocra",
              font_size=14, img_mode="RGBA", color=(0, 0, 0), font_variant="Normal"):
    img = Image.fromarray(np_img, img_mode)
    font = ImageFont.truetype(font_path(font_name), size=font_size)
    # font.set_variation_by_name(font_variant)
    draw = ImageDraw.Draw(img)

    draw.text(xymin, text, font=font, fill=color)
    return np.array(img)


def draw_text_center(np_img: np.ndarray, text: str, ypos=0, font_name: str = "ocra",
                     font_size=14, img_mode="RGBA", color=(0, 0, 0), font_variant="Normal"):
    img = Image.fromarray(np_img, img_mode)
    font = ImageFont.truetype(font_path(font_name), size=font_size)
    # font.set_variation_by_name(font_variant)
    draw = ImageDraw.Draw(img)

    im_height, im_witdh = np_img.shape[:2]
    txt_width, txt_height = draw.textsize(text, font)

    position = (im_witdh - txt_width) / 2, ypos
    draw.text(position, text, font=font, fill=color)
    return np.array(img)


def get_text_position(np_img: np.ndarray, text: str, font_name: str = "ocra",
                      font_size=14, img_mode="RGBA", color=(0, 0, 0), font_variant="Normal"):
    img = Image.fromarray(np_img, img_mode)
    font = ImageFont.truetype(font_path(font_name), size=font_size)
    # font.set_variation_by_name(font_variant)
    draw = ImageDraw.Draw(img)

    im_height, im_witdh = np_img.shape[:2]
    txt_width, txt_height = draw.textsize(text, font)

    position = (im_witdh - txt_width) / 2, (im_height - txt_height) / 2
    # draw.text(position, text, font=font, fill=color)
    return position
