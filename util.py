import numpy as np
import torchvision, torch, os, random
from tqdm import tqdm
from torchvision.transforms import functional as tf


def transform_cv2mod(img_mat):
    res = np.asarray(img_mat, order="C")
    res = np.swapaxes(res, 0, 2)
    res = np.swapaxes(res, 1, 2)
    return res


def gb(img):
    r = random.randint(1, 3) * 2 - 1
    return tf.gaussian_blur(img, kernel_size=r)


def rotate(img):
    r = random.randint(1, 359)
    return tf.rotate(img, r)


def gray(img):
    return tf.rgb_to_grayscale(img, 3)


def rand_aug(input_path, out_path):
    transform = [gb, rotate, tf.vflip, tf.hflip]
    for root, _, files in os.walk(input_path):
        for index, f in tqdm(enumerate(files)):
            t = str(f)[:3]
            img = torchvision.io.read_file(root + f)
            img = torchvision.io.decode_jpeg(img)
            for i in range(3):
                r = random.randint(0, len(transform) - 1)
                p = random.randint(0, 100)
                temp_img = transform[r](img)
                if p < 20:
                    temp_img = gray(temp_img)
                elif 20 <= p <= 40:
                    temp_img = tf.invert(temp_img)
                temp_img = temp_img.cpu()
                temp_img = torchvision.io.encode_jpeg(temp_img)
                torchvision.io.write_file(out_path + f"{t}_modify{3 * index + i}.jpg", temp_img)
