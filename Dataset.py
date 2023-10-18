import os.path
import numpy as np
import cv2 as cv
import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, label_file, image_dirs):
        self.img_label = pd.read_csv(label_file)
        self.img_dir = image_dirs

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_label.iloc[index, 0])
        img = cv.imread(img_path)
        np.asarray(img, order="C")
        img = np.swapaxes(img, 0, 2)
        return img, self.img_label.iloc[index, 1]
    
    def __len__(self):
        return len(self.img_label)
