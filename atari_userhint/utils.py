import cv2 as cv
import numpy as np

from chainer import optimizers


def meanshift_filtering(img, hs=16, hr=64):
    img = cv.pyrMeanShiftFiltering(img, hs, hr)

    return img


def morphology(img, kernel=(5, 5), iteration=1):
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=kernel, iterations=iteration)

    return img


def add_gaussian(img, min_stddev=0, max_stddev=50):
    """Add gaussian noise to the image.
    
    Args:
        img (numpy.uint8): image data that will be noise added.
        min_stddev (int, optional): Defaults to 0. minimum standard deviation.
        max_stddev (int, optional): Defaults to 50. maximum standard deviation.
    
    Returns:
        [numpy.uint8]: [image data added gaussian noise.]
    """

    stddev = 7
    noise = np.random.randn(*img.shape) * stddev

    noised_img = img + noise
    noised_img = np.clip(noised_img, a_min=0, a_max=255).astype(np.uint8)

    return noised_img


def edge_emphasis(img, line_path):
    line = cv.imread(str(line_path))
    line = cv.dilate(line, (5, 5), 2)
    gray_pixels = line < 200
    img[gray_pixels] = 0

    return img


def edge_detection_emphasis(img):
    line = cv.Canny(img, 100, 200)
    gray_pixels = line > 10
    img[gray_pixels] = 0

    return img


def contrast_convert(img, contrast_low=0.5, contrast_high=1.5):
    """the implmentation of contrast conversion
    
    Args:
        img (numpy.uint8): image data
        saturation_low (int, optional): The value of minimum contrast
        saturation_high (int, optional): The value of maximum contrast
    
    Returns:
        [numpy.uint8]: Contrast converted image
    """

    contrast_scale = 0.92
    img = img.astype(float) * contrast_scale
    img[img < 0] = 0
    img[img > 255] = 255
    converted_img = img.astype(np.uint8)

    return converted_img


def gaussian_filter(img):
    """the implementation of gaussian blur
    Args:
        img (numpy.uint8): image data
    Returns:
        numpy.uint8: blurred image data
    """

    filtered_img = cv.GaussianBlur(img, (3, 3), 1)

    return filtered_img


def random_posterize(img, low=1, high=8):
    """the implementation of random posterize
    Args:
        img (numpy.uint8): image data
        low (int, optional): Defaults to 1. The minimum of bit range
        high (int, optional): [description]. The maximum of bit range
    Returns:
        numpy.uint8: posterized image data
    """

    bits = 5
    bits = np.uint8(bits)

    lut = np.arange(0, 256, dtype=np.uint8)
    mask = ~np.uint8(2 ** (8 - bits) - 1)
    lut &= mask

    posterized_img = cv.LUT(img, lut)

    return posterized_img


def kmeans(img, clusters=48):
    """Quantize colors with using kmeans
    Args:
        img (numpy.uint8): image data
        clusters (int, optional): The number of cluters
    Returns:
        numpy.uint8: Quantized image data
    """

    z = img.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv.kmeans(z, clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    quantized_img = center[label.flatten()].reshape((img.shape))

    return quantized_img


def set_optimizer(model, alpha=0.0001, beta=0.9):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer
