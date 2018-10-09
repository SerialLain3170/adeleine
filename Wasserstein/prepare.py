import cv2
import os
import numpy as np

def prepare_dataset_line(filename):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]
        rnd1 = np.random.randint(height+1-256)
        rnd2 = np.random.randint(width+1-256)
        rnd1=1
        rnd2=0

        hr_image = image[rnd1:rnd1+256, rnd2:rnd2+256]

        hr_image = hr_image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image, rnd1, rnd2

def prepare_dataset_line_test(filename):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        hr_image = image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image

def prepare_dataset_color(filename,rnd1,rnd2):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]
        hr_image = image[rnd1:rnd1+256, rnd2:rnd2+256]

        hr_image = hr_image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        return hr_image