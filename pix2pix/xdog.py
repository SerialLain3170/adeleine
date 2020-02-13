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


def xdog_process(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img / 255.0
    sigma = np.random.choice([0.3, 0.4, 0.5])
    img = xdog(img, sigma, 4.5, 19, 0.01, 10^9)
    if np.random.randint(2):
        random_value = np.random.choice([0.0, 0.1, 0.2])
        img[img < 0.9] = random_value

    return img
