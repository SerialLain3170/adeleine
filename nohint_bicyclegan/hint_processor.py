import numpy as np
import cv2 as cv

from typing import List
from typing_extensions import Literal
from xdog import XDoG
from pathlib import Path

LineArt = List[Literal["xdog", "pencil", "digital", "blend"]]


class BasicProtocol:
    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__} will be processed."

    def exec(self, thing):
        NotImplementedError

    def __call__(self, thing):
        return self.exec(thing)


class RandomProtocol(BasicProtocol):
    def __init__(self):
        pass

    def __call__(self, img: np.array) -> np.array:
        if np.random.randint(2):
            return self.exec(img)

        else:
            return img


class AddIntensity(RandomProtocol):
    def __init__(self, intensity=1.7):
        self.intensity = intensity

    def exec(self, img):
        const = 255.0 ** (1.0 - self.intensity)
        img = (const * (img ** self.intensity))

        return img


class Morphology(RandomProtocol):
    def __init__(self):
        self.method = ["erode", "dilate"]

    def exec(self, img):
        method = np.random.choice(self.method)

        if method == "dilate":
            img = cv.dilate(img, (5, 5), iterations=1)
        elif method == "erode":
            img = cv.erode(img, (5, 5), iterations=1)

        return img


class ColorVariant(RandomProtocol):
    def __init__(self, max_value=30, thre=200):
        self.mv = max_value
        self.thre = thre

    def exec(self, img):
        value = np.random.randint(self.mv + 1)
        img[img < self.thre] = value

        return img


class LineSelector(BasicProtocol):
    def __init__(self,
                 sketch_path: Path,
                 line_method: LineArt,
                 blend=0.5):
        self.line_method = line_method
        self.blend = 0.5
        self.pre_intensity = AddIntensity(intensity=1.4)
        self.post_intensity = AddIntensity(intensity=(1/1.5))
        self.sketch_path = sketch_path

        self.xdog_process = XDoG()

        self._message(line_method)

    @staticmethod
    def _message(line_method: LineArt):
        print("Considering these line extraction methods...")
        for method in line_method:
            if method == "xdog":
                print("XDoG will be implemented.")
            elif method == "pencil":
                print("SketchKeras will be implemented.")
            elif method == "blend":
                print("XDoG + SketchKeras will be implemented.")
            elif method == "digital":
                print("Sketch Simplification will be implemented.")
        print("\n")

    def _xdog_preprocess(self, path: Path) -> np.array:
        img = self.xdog_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path: Path) -> np.array:
        filename = path.name
        line_path = self.sketch_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _blend_preprocess(self, path: Path, blend=0.5) -> np.array:
        xdog_line = self._xdog_preprocess(path)
        penc_line = self._pencil_preprocess(path)
        penc_line = self.pre_intensity.exec(penc_line)

        xdog_blur = cv.GaussianBlur(xdog_line, (5, 5), 1)
        xdog_blur = cv.addWeighted(xdog_blur, 0.75, xdog_line, 0.25, 0)

        blend = cv.addWeighted(xdog_blur, blend, penc_line, (1 - blend), 0)

        return self.post_intensity.exec(blend)

    def exec(self, path: Path) -> np.array:
        method = np.random.choice(self.line_method)

        if method == "xdog":
            img = self._xdog_preprocess(path)
        elif method == "pencil":
            img = self._pencil_preprocess(path)
        elif method == "blend":
            img = self._blend_preprocess(path, self.blend)

        return img


class LineProcessor:
    def __init__(self, sketch_path: Path, line_method: LineArt):
        self.process_list = [
            LineSelector(sketch_path, line_method),
            Morphology(),
            ColorVariant()
        ]

        self._message(self.process_list)

    @staticmethod
    def _message(process_list: List):
        print("Considering these augmentations for line arts...")
        for index, process in enumerate(process_list):
            if index > 0:
                print(process)
        print("\n")

    def __call__(self, x: Path) -> np.array:
        for process in self.process_list:
            x = process(x)

        return x
