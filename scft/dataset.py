import torch
import numpy as np
import cv2 as cv

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from xdog import xdog_process
from thin_plate_spline import warping_image
from torchvision.transforms import ColorJitter


class IllustDataset(Dataset):
    """Dataset for training.

       Returns (line, color)
           line: input. Line art of color image
           color: target.
    """
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 extension=".jpg"):

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

        self.sketch_path = sketch_path
        self.src_per = 0.2
        self.tgt_per = 0.05
        self.thre = 50

        self.src_const = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.2, -0.2],
            [-0.2, 0.2],
            [0.2, 0.2],
            [-0.2, -0.2]
        ])

    def _train_val_split(self, pathlist):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    def _coordinate(self, img):
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def _xdog_preprocess(self, path):
        img = xdog_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path):
        filename = path.name
        line_path = self.sketch_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _preprocess(self, path):
        """Returns line art of color image
           This class returns only sketchKeras
        Parameters
        ----------
        path : Path
            Path of color image
        
        Returns
        -------
        img: numpy.array
            Line art of color image
        """
        #method = np.random.choice([pencil])

        #if method == "xdog":
        #    img = self._xdog_preprocess(path)
        #elif method == "pencil":
        img = self._pencil_preprocess(path)

        return img

    def _warp(self, img):
        const = self.src_const
        c_src = const + np.random.uniform(-self.src_per, self.src_per, (8, 2))
        c_tgt = c_src + np.random.uniform(-self.tgt_per, self.tgt_per, (8, 2))

        img = warping_image(img, c_src, c_tgt)

        return img

    def _jitter(self, img):
        img = img.astype(np.float32)
        noise = np.random.uniform(-self.thre, self.thre)
        img += noise
        img = np.clip(img, 0, 255)

        return img

    def valid(self, validsize):
        c_valid_box = []
        l_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            color = cv.imread(str(color_path))
            line = self._preprocess(color_path)

            jitter = self._jitter(color)
            warp = self._warp(jitter)

            color = self._coordinate(warp)
            line = self._coordinate(line)

            c_valid_box.append(color)
            l_valid_box.append(line)

        color = self._totensor(c_valid_box)
        line = self._totensor(l_valid_box)

        return color, line

    @staticmethod
    def _totensor(array_list):
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        color_path = self.train_list[idx]
        color = cv.imread(str(color_path))
        line = self._preprocess(color_path)

        return color, line


class IllustTestDataset(Dataset):
    """Dataset for inference/test.

       Returns (line_path, color_path)
           line_path: path of line art
           color_path: path of color image
    """
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path):

        self.path = data_path
        self.pathlist = list(self.path.glob('**/*.png'))
        self.pathlen = len(self.pathlist)

        self.sketch_path = sketch_path
        self.test_len = 200

    def __repr__(self):
        return f"dataset length: {self.pathlen}"

    def __len__(self):
        return self.test_len

    def __getitem__(self, idx):
        line_path = self.pathlist[idx]
        line_path = self.sketch_path / line_path.name

        rnd = np.random.randint(self.pathlen)
        style_path = self.pathlist[rnd]

        return line_path, style_path


class LineCollator:
    """Collator for training.
    """
    def __init__(self,
                 img_size=224,
                 src_perturbation=0.2,
                 dst_perturbation=0.05,
                 brightness=0.3,
                 contrast=0.5,
                 saturation=0.1,
                 hue=0.3):

        self.size = img_size
        self.src_per = src_perturbation
        self.tgt_per = dst_perturbation
        self.thre = 50

        self.src_const = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.2, -0.2],
            [-0.2, 0.2],
            [0.2, 0.2],
            [-0.2, -0.2]
        ])

        self.jittering = ColorJitter(brightness, contrast, saturation, hue)

    @staticmethod
    def _random_crop(line, color, size):
        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    @staticmethod
    def _coordinate(image):
        """3 stage of manipulation
           - BGR -> RGB
           - (H, W, C) -> (C, H, W)
           - Normalize
        
        Parameters
        ----------
        image : numpy.array
            image data
        
        Returns
        -------
        numpy.array
            manipulated image data
        """
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _warp(self, img):
        """Spatial augment by TPS
        """
        const = self.src_const
        c_src = const + np.random.uniform(-self.src_per, self.src_per, (8, 2))
        c_tgt = c_src + np.random.uniform(-self.tgt_per, self.tgt_per, (8, 2))

        img = warping_image(img, c_src, c_tgt)

        return img

    def _jitter(self, img):
        """Color augment
        """
        img = img.astype(np.float32)
        noise = np.random.uniform(-self.thre, self.thre)
        img += noise
        img = np.clip(img, 0, 255)

        return img

    def _prepair(self, color, line):
        """3 stages of preparation
           - Crop
           - Spatial & Color augmentation
           - Coordination
        """
        line, color = self._random_crop(line, color, size=self.size)

        jittered = self._jitter(color)
        warped = self._warp(jittered)

        jittered = self._coordinate(jittered)
        warped = self._coordinate(warped)
        line = self._coordinate(line)

        return jittered, warped, line

    def __call__(self, batch):
        j_box = []
        w_box = []
        l_box = []

        for b in batch:
            color, line = b
            jitter, warped, line = self._prepair(color, line)

            j_box.append(jitter)
            w_box.append(warped)
            l_box.append(line)

        j = self._totensor(j_box)
        w = self._totensor(w_box)
        l = self._totensor(l_box)

        return (j, w, l)


class LineTestCollator:
    """Collator for inference/test.
    """
    def __init__(self):
        pass

    @staticmethod
    def _coordinate(image):
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _prepare(self, image_path, style_path, size=512):
        line = cv.imread(str(image_path))
        line = cv.resize(line, (192, 256), interpolation=cv.INTER_CUBIC)
        color = cv.imread(str(style_path))

        color = self._coordinate(color)
        line = self._coordinate(line)

        return color, line

    def __call__(self, batch):
        c_box = []
        l_box = []

        for bpath, style in batch:
            color, line = self._prepare(bpath, style)

            c_box.append(color)
            l_box.append(line)

        c = self._totensor(c_box)
        l = self._totensor(l_box)

        return (c, l)