import cv2
import os
import numpy as np

def sharpImage(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv2.GaussianBlur(img,(0, 0), sigma)
    G_large = cv2.GaussianBlur(img,(0, 0), sigma_large)
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

def prepare_dataset_line(filename,size=256):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]
        rnd1 = np.random.randint(height+1-size)
        rnd2 = np.random.randint(width+1-size)

        hr_image = image[rnd1:rnd1+size, rnd2:rnd2+size]

        hr_image = hr_image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

def prepare_dataset_color(filename,rnd1,rnd2,size=256):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]
        hr_image = image[rnd1:rnd1+size, rnd2:rnd2+size]

        hr_image = hr_image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image