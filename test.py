import torch
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv.imread("./input/test/test.jpg")
    img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)

    m = torch.load("save/resnet50d.m").cpu()
    m.eval()
    np.asarray(img, order="C")
    img = np.swapaxes(img, 0, 2)
    img = torch.Tensor(img).unsqueeze(0)

    y = "dog" if m(img)[0][0] > 0.5 else "cat"
    print(y)


