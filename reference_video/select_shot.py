import shutil
import argparse
import cv2 as cv
import numpy as np

from pathlib import Path


def calc_hist(img: np.array) -> np.array:
    hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv.calcHist([img], [2], None, [256], [0, 256])
    hist = np.concatenate([hist_b, hist_g, hist_r], axis=0)

    return hist[:, 0]


def calc_difference(anime_dir: Path,
                    thre,
                    min_frames=8):
    f = open(f"{anime_dir}/separate.txt", "w")
    pathlist = list(anime_dir.glob("*.png"))
    nums = len(pathlist)
    start_num = 1
    for num in range(1, nums):
        img = cv.imread(str(f"{anime_dir}/{num}.png"))
        hist = calc_hist(img)
        img_1 = cv.imread(str(f"{anime_dir}/{num + 1}.png"))
        hist_1 = calc_hist(img_1)

        diff = (np.abs(hist_1 - hist)).mean()

        if diff > thre:
            if num - start_num > min_frames:
                print(num, start_num)
                f.write(f"{num},{start_num}\n")

            start_num = num + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select scene")
    parser.add_argument("--d", type=Path, help="Directory that contains anime frames")
    parser.add_argument("--th", type=int, help="Threshold that indicates transition of the scenes")
    args = parser.parse_args()

    calc_difference(args.d, args.th)
