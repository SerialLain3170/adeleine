import numpy as np
import cv2 as cv

from PIL import Image, ImageOps, ImageFilter
from pathlib import Path


def sharpImage(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv.GaussianBlur(img, (0, 0), sigma)
    G_large = cv.GaussianBlur(img, (0, 0), sigma_large)
    S = (1+p) * G_small - p * G_large

    return S


def softThreshold(SI, epsilon, phi):
    T = np.zeros(SI.shape)
    SI_bright = SI >= epsilon
    SI_dark = SI < epsilon
    T[SI_bright] = 1.0
    T[SI_dark] = 1.0 + np.tanh(phi * (SI[SI_dark] - epsilon))

    return T


def xdog(img, sigma, k_sigma, p, epsilon, phi):
    S = sharpImage(img, sigma, k_sigma, p)
    SI = np.multiply(img, S)
    T = softThreshold(SI, epsilon, phi)

    return T


def line_process(filename):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img / 255.0
    sigma = np.random.choice([0.3, 0.4, 0.5])
    img = xdog(img, sigma, 4.5, 19, 0.01, 10^9)
    if np.random.randint(2):
        random_value = np.random.choice([0.0, 0.1, 0.2])
        img[img < 0.9] = random_value

    return img


def sketch_process(filename):
    img = Image.open(filename)
    img = ImageOps.grayscale(img)
    img = img.filter(ImageFilter.FIND_EDGES)
    img = img.filter(ImageFilter.SMOOTH)
    img = ImageOps.invert(img)
    img.save("test.png")


def line_example_process(filename):
    img = cv.imread(filename)
    #img[img < 11] = 255
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img / 255.0
    sigma = np.random.choice([0.3, 0.4, 0.5])
    img = xdog(img, sigma, 4.5, 19, 0.01, 10^9)
    img = img * 255
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.tile(img, (1, 1, 3))

    return img


if __name__ == "__main__":
    img_path = "./job_kasyu.png"
    sketch_process(img_path)
    #img = line_example_process(img_path)
    #img[img < 200] = 0
    #cv.imwrite("./0.png", img)