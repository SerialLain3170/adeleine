import numpy as np
import random
import cv2 as cv
import copy
import chainer
import chainer.functions as F

from xdog import line_process
from chainer import cuda
from pathlib import Path
from scipy.stats import truncnorm
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool

xp = cuda.cupy
cuda.get_device(0).use()


class DataLoader:
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 digi_path: Path,
                 paint_type="cell",
                 interpolate=False,
                 extension=".jpg",
                 img_size=224):

        self.data_path = data_path
        self.sketch_path = sketch_path
        self.digi_path = digi_path

        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)

        self.size = img_size
        self.paint_type = paint_type
        self.interpolate = interpolate
        self.interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )

    def __str__(self):
        return f"dataset path: {self.data_path} dataset length: {self.train_len}"

    # Initialization method
    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.95)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    # Line art preparation method
    @staticmethod
    def _add_intensity(img, intensity=1.7):
        const = 255.0 ** (1.0 - intensity)
        img = (const * (img ** intensity))

        return img

    @staticmethod
    def _morphology(img):
        method = np.random.choice(["dilate", "erode"])
        if method == "dilate":
            img = cv.dilate(img, (5, 5), iterations=1)
        elif method == "erode":
            img = cv.erode(img, (5, 5), iterations=1)

        return img

    @staticmethod
    def _color_variant(img, max_value=30):
        color = np.random.randint(max_value + 1)
        img[img < 200] = color

        return img

    def _detail_preprocess(self, img):
        intensity = np.random.randint(2)
        morphology = np.random.randint(2)
        color_variance = np.random.randint(2)

        if intensity:
            img = self._add_intensity(img)
        if morphology:
            img = self._morphology(img)
        if color_variance:
            img = self._color_variant(img)

        return img

    def _xdog_preprocess(self, path):
        img = line_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path):
        filename = path.name
        line_path = self.sketch_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _digital_preprocess(self, path):
        filename = path.name
        line_path = self.digi_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _blend_preprocess(self, path, blend=0.5):
        xdog_line = self._xdog_preprocess(path)
        penc_line = self._pencil_preprocess(path)
        penc_line = self._add_intensity(penc_line, 1.4)

        xdog_blur = cv.GaussianBlur(xdog_line, (5, 5), 1)
        xdog_blur = cv.addWeighted(xdog_blur, 0.75, xdog_line, 0.25, 0)

        blend = cv.addWeighted(xdog_blur, blend, penc_line, (1 - blend), 0)

        return self._add_intensity(blend, (1/1.5))

    def _preprocess(self, path):
        method = np.random.choice(["xdog", "pencil", "digital", "blend"])

        if method == "xdog":
            img = self._xdog_preprocess(path)
        elif method == "pencil":
            img = self._pencil_preprocess(path)
        elif method == "digital":
            img = self._digital_preprocess(path)
        elif method == "blend":
            img = self._blend_preprocess(path)

        img = self._detail_preprocess(img)

        return img

    # Preprocess method
    @staticmethod
    def _random_crop(line, color, size):
        scale = np.random.randint(288, 768)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    @staticmethod
    def _coordinate(image):
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _variable(image_list):
        return chainer.as_variable(xp.array(image_list).astype(xp.float32))

    # Hint preparation method
    @staticmethod
    def _making_mask(mask, color, size):
        choice = np.random.choice(['width', 'height', 'diag'])

        if choice == 'width':
            rnd_height = np.random.randint(4, 8)
            rnd_width = np.random.randint(4, 64)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'height':
            rnd_height = np.random.randint(4, 64)
            rnd_width = np.random.randint(4, 8)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'diag':
            rnd_height = np.random.randint(4, 8)
            rnd_width = np.random.randint(4, 64)

            rnd1 = np.random.randint(size - rnd_height - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            for index in range(rnd_width):
                mask[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index]

        return mask

    def _prepare_pair(self, image_path, size):
        interpolation = random.choice(self.interpolations)
        color = cv.imread(str(image_path))
        line = self._preprocess(image_path)
        line, color = self._random_crop(line, color, size=size)

        # Hint preparation
        mask = copy.copy(line)
        repeat = np.random.randint(8, 20)
        for _ in range(repeat):
            mask = self._making_mask(mask, color, size=size)
        mask_ds = cv.resize(mask, (int(size/2), int(size/2)), interpolation=interpolation)

        color = self._coordinate(color)
        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (color, line, mask, mask_ds)

    @staticmethod
    def _interpolate(img):
        height, width = img.shape[0], img.shape[1]
        canvas = np.zeros(shape=(1024, 784, 3), dtype=np.uint8)
        canvas[:height, :width, :] = img

        return canvas

    def _prepare_test(self, line_path, mask_path):
        print(f"line path: {line_path}")
        print(f"mask path: {mask_path}")
        line = cv.imread(str(line_path))
        mask = cv.imread(str(mask_path))
        if self.interpolate:
            line = self._interpolate(line)
            mask = self._interpolate(mask)
        if self.paint_type == "imp":
            line = cv.morphologyEx(line, cv.MORPH_CLOSE, (5, 5), iterations=2)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (5, 5), iterations=2)
        height, width = line.shape[0], line.shape[1]
        mask_ds = cv.resize(mask, (int(width/2), int(height/2)))

        print(line.shape, mask.shape, mask_ds.shape)

        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (line, mask, mask_ds)

    def test(self, line_path, mask_path):
        line_box = []
        mask_box = []
        mask_ds_box = []

        line, mask, mask_ds = self._prepare_test(line_path, mask_path)
        line_box.append(line)
        mask_box.append(mask)
        mask_ds_box.append(mask_ds)

        line = self._variable(line_box)
        mask = self._variable(mask_box)
        mask_ds = self._variable(mask_ds_box)

        return (line, mask, mask_ds)

    def __call__(self, batchsize, mode='train'):
        color_box = []
        line_box = []
        mask_box = []
        mask_ds_box = []

        for _ in range(batchsize):
            if mode == 'train':
                rnd = np.random.randint(self.train_len)
                image_path = self.train[rnd]
            elif mode == 'valid':
                rnd = np.random.randint(self.valid_len)
                image_path = self.valid[rnd]
            else:
                raise AttributeError

            color, line, mask, mask_ds = self._prepare_pair(image_path, size=self.size)

            color_box.append(color)
            line_box.append(line)
            mask_box.append(mask)
            mask_ds_box.append(mask_ds)

        color = self._variable(color_box)
        line = self._variable(line_box)
        mask = self._variable(mask_box)
        mask_ds = self._variable(mask_ds_box)

        return (color, line, mask, mask_ds)


class RefineDataset:
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 digi_path: Path,
                 st_path: Path,
                 paint_type="cell",
                 interpolate=False,
                 extension=".jpg",
                 img_size=512,
                 crop_size=256):

        self.data_path = data_path
        self.sketch_path = sketch_path
        self.digi_path = digi_path
        self.st_path = st_path

        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)

        self.paint_type = paint_type
        self.interpolate = interpolate
        self.img_size = img_size
        self.size = crop_size
        self.spray_split = 7
        self.interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )

    def __str__(self):
        return f"dataset path: {self.data_path} dataset length: {self.train_len}"

    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.95)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    # Line art preparation method
    @staticmethod
    def _add_intensity(img, intensity=1.7):
        const = 255.0 ** (1.0 - intensity)
        img = (const * (img ** intensity))

        return img

    @staticmethod
    def _morphology(img):
        method = np.random.choice(["dilate", "erode"])
        if method == "dilate":
            img = cv.dilate(img, (5, 5), iterations=1)
        elif method == "erode":
            img = cv.erode(img, (5, 5), iterations=1)

        return img

    @staticmethod
    def _color_variant(img, max_value=30):
        color = np.random.randint(max_value + 1)
        img[img < 200] = color

        return img

    def _detail_preprocess(self, img):
        intensity = np.random.randint(2)
        morphology = np.random.randint(2)
        color_variance = np.random.randint(2)

        if intensity:
            img = self._add_intensity(img)
        if morphology:
            img = self._morphology(img)
        if color_variance:
            img = self._color_variant(img)

        return img

    def _xdog_preprocess(self, path):
        img = line_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path):
        filename = path.name
        line_path = self.sketch_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _digital_preprocess(self, path):
        filename = path.name
        line_path = self.digi_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _blend_preprocess(self, path, blend=0.5):
        xdog_line = self._xdog_preprocess(path)
        penc_line = self._pencil_preprocess(path)
        penc_line = self._add_intensity(penc_line, 1.4)

        xdog_blur = cv.GaussianBlur(xdog_line, (5, 5), 1)
        xdog_blur = cv.addWeighted(xdog_blur, 0.75, xdog_line, 0.25, 0)

        blend = cv.addWeighted(xdog_blur, blend, penc_line, (1 - blend), 0)

        return self._add_intensity(blend, (1/1.5))

    def _preprocess(self, path):
        method = np.random.choice(["xdog", "pencil", "digital", "blend"])

        if method == "xdog":
            img = self._xdog_preprocess(path)
        elif method == "pencil":
            img = self._pencil_preprocess(path)
        elif method == "digital":
            img = self._digital_preprocess(path)
        elif method == "blend":
            img = self._blend_preprocess(path)

        img = self._detail_preprocess(img)

        return img

    # Preprocess method
    @staticmethod
    def _random_crop(line, color, color_mask, size):
        scale = np.random.randint(280, 768)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))
        color_mask = cv.resize(color_mask, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color_mask = color_mask[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color, color_mask

    def _random_resize(self, line, color):
        line = cv.resize(line, (self.size, self.size))
        color = cv.resize(color, (self.size, self.size))

        return line, color

    @staticmethod
    def _coordinate(image):
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _variable(image_list):
        return chainer.as_variable(xp.array(image_list).astype(xp.float32))

    # Hint preparation method
    @staticmethod
    def _making_mask(mask, color, size):
        choice = np.random.choice(['width', 'height', 'diag'])

        if choice == 'width':
            rnd_height = np.random.randint(8, 16)
            rnd_width = np.random.randint(8, 64)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'height':
            rnd_height = np.random.randint(8, 64)
            rnd_width = np.random.randint(8, 16)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'diag':
            rnd_height = np.random.randint(8, 16)
            rnd_width = np.random.randint(8, 64)

            rnd1 = np.random.randint(size - rnd_height - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            for index in range(rnd_width):
                mask[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index]

        return mask

    # Simulation method
    def _overlap_crop(self, color, size):
        #color, _, _ = self._random_crop(color, color, color)
        rnd1 = np.random.randint(500 - size)
        rnd2 = np.random.randint(500 - size)

        return color[rnd1: rnd1 + size, rnd2: rnd2 + size]

    def _overlap(self, color, color_crop, size):
        rnd1 = np.random.randint(500 - size)
        rnd2 = np.random.randint(500 - size)
        color[rnd1: rnd1 + size, rnd2: rnd2 + size] = color_crop

        return color

    def _making_overlap(self, color, size, iteration=7):
        for _ in range(iteration):
            image_path = self.train[np.random.randint(self.train_len)]
            overlap_color = cv.imread(str(image_path))
            size = np.random.randint(100, 200)
            color_crop = self._overlap_crop(overlap_color, size)
            color = self._overlap(color, color_crop, size=size)

        return color

    @staticmethod
    def _min_dis(point, point_list):
        dis = []
        for p in point_list:
            dis.append(np.sqrt(np.sum(np.square(np.array(point)-np.array(p)))))

        return min(dis)

    @staticmethod
    def _get_dominant_color(image):
        image = image.convert('RGBA')
        image.thumbnail((200, 200))

        max_score = 0
        dominant_color = 0

        for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
            if a == 0:
                continue

            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            if y > 0.9:
                continue
            if ((r > 230) & (g > 230) & (b > 230)) or ((r < 30) & (g < 30) & (b < 30)):
                continue
            # Calculate the score, preferring highly saturated colors.
            # Add 0.1 to the saturation so we don't completely ignore grayscale
            # colors by multiplying the count by zero, but still give them a low
            # weight.
            score = (saturation + 0.1) * count
            if score > max_score:
                max_score = score
                dominant_color = (r, g, b)

        return dominant_color

    def _spray(self, color_img, iteration=2, eps=1e-8):
        # To get dominant color, conversion array to pillow object
        img_pillow = Image.fromarray(color_img)
        color = self._get_dominant_color(img_pillow)

        # To process this method, conversion pillow object to array
        img = np.array(color_img)
        h = int(self.size/self.spray_split)
        w = int(self.size/self.spray_split)
        a_x = np.random.randint(0, h)
        a_y = np.random.randint(0, w)
        b_x = np.random.randint(0, h)
        b_y = np.random.randint(0, w)
        begin_point = np.array([min(a_x, b_x), a_y])
        end_point = np.array([max(a_x, b_x), b_y])
        tan = (begin_point[1] - end_point[1]) / (begin_point[0] - end_point[0] + 0.001)

        center_point_list = []
        for i in range(begin_point[0], end_point[0]+1):
            a = i
            b = (i-begin_point[0])*tan + begin_point[1]
            center_point_list.append(np.array([int(a), int(b)]))
        center_point_list = np.array(center_point_list)

        lamda = random.uniform(0.01, 10)
        paper = np.zeros((h, w, 3))
        mask = np.zeros((h, w))
        center = [int(h/2), int(w/2)]
        paper[center[0], center[1], :] = color
        for i in range(h):
            for j in range(w):
                dis = self._min_dis([i, j], center_point_list)
                paper[i, j, :] = np.array(color)/(np.exp(lamda*dis) + eps)
                mask[i, j] = np.array([255])/(np.exp(lamda*dis) + eps)

        paper = (paper).astype('uint8')
        mask = (mask).astype('uint8')

        mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
        im = cv.resize(paper, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)

        # To implement paste method, conversion array to pillow object
        imq = Image.fromarray(im)
        imp = img_pillow.copy()

        imp.paste(imq, (0, 0, imp.size[0], imp.size[1]), mask=Image.fromarray(mask))
        imp = np.asarray(imp)

        return imp

    def _spatial_transformer(self, image_path):
        filename = image_path.name
        filepath = self.st_path / Path(filename)
        st_img = cv.imread(str(filepath))

        while st_img is None:
            print(f"Spatial Transformer image is None !")
            image_path = self.train[np.random.randint(self.train_len)]
            st_img = cv.imread(str(image_path))

        st_img = cv.resize(st_img, (self.img_size, self.img_size))

        return st_img

    def _simulate(self, image_path):
        st_img = self._spatial_transformer(image_path)
        st_img = self._making_overlap(st_img, size=self.img_size, iteration=3)
        st_img = self._spray(st_img)

        return st_img

    def _prepare_pair(self, image_path, size):
        interpolation = random.choice(self.interpolations)
        color = cv.imread(str(image_path))
        line = self._preprocess(image_path)
        color_mask = self._simulate(image_path)
        line, color, color_mask = self._random_crop(line, color, color_mask, size=size)

        # Hint prepration
        mask = copy.copy(line)
        repeat = np.random.randint(8, 20)
        for _ in range(repeat):
            mask = self._making_mask(mask, color, size=size)
        mask_ds = cv.resize(mask, (int(size/2), int(size/2)), interpolation=interpolation)

        color = self._coordinate(color)
        color_mask = self._coordinate(color_mask)
        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (color, line, mask, mask_ds, color_mask)

    @staticmethod
    def _interpolate(img):
        height, width = img.shape[0], img.shape[1]
        canvas = np.zeros(shape=(1024, 784, 3), dtype=np.uint8)
        canvas[:height, :width, :] = img

        return canvas

    def _prepare_test(self, line_path, mask_path):
        print(f"line path: {line_path}")
        print(f"mask path: {mask_path}")
        line = cv.imread(str(line_path))
        mask = cv.imread(str(mask_path))
        if self.interpolate:
            line = self._interpolate(line)
            mask = self._interpolate(mask)
        if self.paint_type == "imp":
            line = cv.morphologyEx(line, cv.MORPH_CLOSE, (5, 5), iterations=2)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (5, 5), iterations=2)
        height, width = line.shape[0], line.shape[1]
        mask_ds = cv.resize(mask, (int(width/2), int(height/2)))

        print(line.shape, mask.shape, mask_ds.shape)

        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (line, mask, mask_ds)

    def test(self, line_path, mask_path):
        line_box = []
        mask_box = []
        mask_ds_box = []

        line, mask, mask_ds = self._prepare_test(line_path, mask_path)
        line_box.append(line)
        mask_box.append(mask)
        mask_ds_box.append(mask_ds)

        line = self._variable(line_box)
        mask = self._variable(mask_box)
        mask_ds = self._variable(mask_ds_box)

        return (line, mask, mask_ds)

    def __call__(self, batchsize, mode='train'):
        color_box = []
        line_box = []
        mask_box = []
        mask_ds_box = []
        cm_box = []

        for _ in range(batchsize):
            if mode == 'train':
                rnd = np.random.randint(self.train_len)
                image_path = self.train[rnd]
            elif mode == 'valid':
                rnd = np.random.randint(self.valid_len)
                image_path = self.valid[rnd]
            else:
                raise AttributeError

            color, line, mask, mask_ds, color_mask = self._prepare_pair(image_path, size=self.size)

            color_box.append(color)
            line_box.append(line)
            mask_box.append(mask)
            mask_ds_box.append(mask_ds)
            cm_box.append(color_mask)

        color = self._variable(color_box)
        line = self._variable(line_box)
        mask = self._variable(mask_box)
        mask_ds = self._variable(mask_ds_box)
        color_mask = self._variable(cm_box)

        return (color, line, mask, mask_ds, color_mask)


if __name__ == "__main__":
    path = Path('./danbooru-images/')
    dataset = RefineDataset(path)
    dataset.test()
