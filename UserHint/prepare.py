import cv2
import os
import numpy as np
import copy
from refine_augmentation import spray, transform, non_affine

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

def making_mask(line_mask, color):
    choice = np.random.choice(['width', 'height', 'diag'])

    if choice == 'width':
        rnd_height = np.random.randint(4, 8)
        rnd_width = np.random.randint(4, 64)

        rnd1 = np.random.randint(224 - rnd_height)
        rnd2 = np.random.randint(224 - rnd_width)
        line_mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

    elif choice == 'height':
        rnd_height = np.random.randint(4, 64)
        rnd_width = np.random.randint(4, 8)

        rnd1 = np.random.randint(224 - rnd_height)
        rnd2 = np.random.randint(224 - rnd_width)
        line_mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

    elif choice == 'diag':
        rnd_height = np.random.randint(4, 8)
        rnd_width = np.random.randint(4, 64)

        rnd1 = np.random.randint(224 - rnd_height - rnd_width - 1)
        rnd2 = np.random.randint(224 - rnd_width)

        for index in range(rnd_width):
            line_mask[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index]

    return line_mask

def prepare_dataset(line_path, color_path, size=224):
    line = cv2.imread(line_path)
    color = cv2.imread(color_path)
    if not line is None and color is not None:
        height, width = line.shape[0], line.shape[1]
        rnd1 = np.random.randint(height+1-size)
        rnd2 = np.random.randint(width+1-size)
        line = line[rnd1:rnd1+size, rnd2:rnd2+size]
        color = color[rnd1:rnd1+size, rnd2:rnd2+size]

        line_mask = copy.copy(line)
        for _ in range(16):
            line_mask = making_mask(line_mask, color)

        line_mask_ds = cv2.resize(line_mask, (int(size/2), int(size/2)))

        color = color[:,:,::-1]
        color = color.transpose(2,0,1)
        color = (color-127.5)/127.5

        line = line[:,:,::-1]
        line = line.transpose(2,0,1)
        line = (line - 127.5)/127.5

        line_mask = line_mask[:,:,::-1]
        line_mask = line_mask.transpose(2,0,1)
        line_mask = (line_mask) / 255.0

        line_mask_ds = line_mask_ds[:,:,::-1]
        line_mask_ds = line_mask_ds.transpose(2,0,1)
        line_mask_ds = (line_mask_ds) / 255.0

        return color, line, line_mask, line_mask_ds

def prepare_refine(line_path, color_path, size=224):
    line = cv2.imread(line_path)
    color = cv2.imread(color_path)
    if not line is None and color is not None:
        height, width = line.shape[0], line.shape[1]
        rnd1_orig = np.random.randint(height+1-size)
        rnd2_orig = np.random.randint(width+1-size)
        line = line[rnd1_orig:rnd1_orig+size, rnd2_orig:rnd2_orig+size]
        color = color[rnd1_orig:rnd1_orig+size, rnd2_orig:rnd2_orig+size]

        color_ds = cv2.resize(color, (int(size/2), int(size/2)))
        line_ds = cv2.resize(line, (int(size/2), int(size/2)))

        mask = np.zeros((int(size/2),int(size/2),1))
        line_mask = copy.copy(line_ds)
        for _ in range(24):
            rnd1 = np.random.randint(int(size/2))
            rnd2 = np.random.randint(int(size/2))
            mask[rnd1: rnd1+2, rnd2:rnd2+10] = 1.0
            line_mask[rnd1:rnd1+2, rnd2:rnd2+10] = 0.0
            line_mask[rnd1:rnd1+2, rnd2:rnd2+10] = color_ds[rnd1:rnd1+2, rnd2:rnd2+10]

        color = color[:,:,::-1]
        color = color.transpose(2,0,1)
        color = (color-127.5)/127.5

        choice = np.random.choice(['spray', 'transform','nonaffine'])
        if choice == 'spray':
            line = spray(color_path, rnd1_orig, rnd2_orig)
        elif choice == 'transform':
            line = transform(color_path, rnd1_orig, rnd2_orig)
        elif choice == 'nonaffine':
            line = non_affine(color_path, rnd1_orig, rnd2_orig)

        mask = mask.transpose(2,0,1)

        line_mask = line_mask[:,:,::-1]
        line_mask = line_mask.transpose(2,0,1)
        line_mask = (line_mask) / 255.0

        return color, line, line_mask, mask

def prepare_test(line_path, mask_path, size=224):
    line = cv2.imread(line_path)
    line_mask = cv2.imread(mask_path)
    if not line is None and line_mask is not None:
        height, width = line.shape[0], line.shape[1]
        #rnd1 = np.random.randint(height+1-size)
        #rnd2 = np.random.randint(width+1-size)
        #line = line[rnd1:rnd1+size, rnd2:rnd2+size]
        #color = color[rnd1:rnd1+size, rnd2:rnd2+size]

        #color_ds = cv2.resize(color, (int(size/2), int(size/2)))
        #line_ds = cv2.resize(line, (int(size/2), int(size/2)))

        line_mask_ds = cv2.resize(line_mask, (128,128))

        line = line[:,:,::-1]
        line = line.transpose(2,0,1)
        line = (line - 127.5)/127.5

        line_mask = line_mask[:,:,::-1]
        line_mask = line_mask.transpose(2,0,1)
        line_mask = (line_mask) / 255.0

        line_mask_ds = line_mask_ds[:,:,::-1]
        line_mask_ds = line_mask_ds.transpose(2,0,1)
        line_mask_ds = (line_mask_ds) / 255.0

        return line, line_mask, line_mask_ds