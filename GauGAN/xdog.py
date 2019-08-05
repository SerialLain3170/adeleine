import numpy as np
import cv2 as cv

def sharpImage(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv.GaussianBlur(img,(0, 0), sigma)
    G_large = cv.GaussianBlur(img,(0, 0), sigma_large)
    S = (1+p) * G_small - p * G_large

    return S

def softThreshold(SI, epsilon, phi):
    T = np.zeros(SI.shape)
    SI_bright = SI >= epsilon
    SI_dark = SI < epsilon
    T[SI_bright] = 1.0
    T[SI_dark] = 1.0 + np.tanh( phi * (SI[SI_dark] - epsilon))

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
    img = xdog(img, sigma, 4.5, 19,0.01, 10^9)
    img = img * 255
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.tile(img, (1,1,3))

    return img