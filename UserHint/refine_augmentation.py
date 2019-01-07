import cv2 as cv
import numpy as np
import chainer.functions as F

def preprocess(image):
    image = image[:, :, ::-1]
    image = image.transpose(2,0,1)
    image = (image - 127.5) / 127.5

    return image

def spray(filename, rnd1, rnd2):
    image = cv.imread(filename)
    if image is not None:
        image = image[rnd1:rnd1+224, rnd2:rnd2+224]
        height, width = image.shape[0], image.shape[1]

        spray_width = np.random.randint(64, 128)
        spray_height = np.random.randint(64,128)

        rnd1 = np.random.randint(height - spray_height)
        rnd2 = np.random.randint(width - spray_width)

        R = np.random.randint(1,255)
        G = np.random.randint(1,255)
        B = np.random.randint(1,255)

        image[rnd1 : rnd1 + spray_height, rnd2 : rnd2 + spray_width, 0] = B * np.ones((spray_height, spray_width))
        image[rnd1 : rnd1 + spray_height, rnd2 : rnd2 + spray_width, 1] = G * np.ones((spray_height, spray_width))
        image[rnd1 : rnd1 + spray_height, rnd2 : rnd2 + spray_width, 2] = R * np.ones((spray_height, spray_width))

        img = preprocess(image)

        return img

def transform(filename, rnd1, rnd2):
    image = cv.imread(filename)
    if image is not None:
        image = image[rnd1:rnd1+224, rnd2:rnd2+224]
        height, width = image.shape[0], image.shape[1]

        c_rnd1 = np.random.randint(height - 64)
        c_rnd2 = np.random.randint(width - 64)
        crop1 = image[c_rnd1 : c_rnd1+64, c_rnd2 : c_rnd2 + 64]
        crop1 = cv.resize(crop1, (96, 16))

        r_rnd1 = np.random.randint(height - 16)
        r_rnd2 = np.random.randint(width - 96)
        image[r_rnd1 : r_rnd1 + 16, r_rnd2 : r_rnd2 + 96, :] = crop1

        c_rnd1 = np.random.randint(height - 64)
        c_rnd2 = np.random.randint(width - 64)
        crop2 = image[c_rnd1 : c_rnd1+64, c_rnd2 : c_rnd2 + 64]
        crop2 = cv.resize(crop1, (16, 96))

        r_rnd1 = np.random.randint(height - 96)
        r_rnd2 = np.random.randint(width - 16)
        image[r_rnd1 : r_rnd1 + 96, r_rnd2 : r_rnd2 + 16, :] = crop2

        c_rnd1 = np.random.randint(height - 64)
        c_rnd2 = np.random.randint(width - 64)
        crop3 = image[c_rnd1 : c_rnd1+64, c_rnd2 : c_rnd2 + 64]
        crop3 = cv.resize(crop1, (112, 32))

        r_rnd1 = np.random.randint(height - 32)
        r_rnd2 = np.random.randint(width - 112)
        image[r_rnd1 : r_rnd1 + 32, r_rnd2 : r_rnd2 + 112, :] = crop3

        img = preprocess(image)

        return img

def non_affine(filename, rnd1, rnd2):
    image = cv.imread(filename)
    if image is not None:
        image = image[rnd1:rnd1+224, rnd2:rnd2+224]
        height, width = image.shape[0], image.shape[1]

        image = preprocess(image)
        image = image.reshape(1,3,224,224).astype(np.float32)
        grid = np.random.uniform(0, 1, size=(1,2,224,224)).astype(np.float32)
        img = F.spatial_transformer_sampler(image, grid)

        img = img[0].data

        return img