import numpy as np
import torchvision, torch, os, random
from tqdm import tqdm
from torchvision.transforms import functional as tf
import csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def rand_aug(input_path, out_path=None):
    """
    Apply random augmentations to images in the input directory and save the modified images to the output directory.

    Args:
        input_path (str): The path to the input directory containing the images.
        out_path (str, optional): The path to the output directory where the modified images will be saved. If not provided, the modified images will be saved in the same directory as the input images.

    Returns:
        None
    """
    transform = [gb, rotate, tf.vflip, tf.hflip]
    out_path = out_path if out_path else input_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for root, _, files in os.walk(input_path):
        for index, f in enumerate(tqdm(files)):
            if not f.endswith(('.jpg', '.jpeg', '.png')):
                continue
            t = str(f)[:3]
            img = torchvision.io.read_file(os.path.join(root, f))
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
                torchvision.io.write_file(os.path.join(out_path, f"{t}_modify{3 * index + i}.jpg"), temp_img)


def generate_csv(data_folder_path):
    """
    Generate a CSV file containing image names and corresponding labels.

    Args:
        data_folder_path (str): The path to the folder containing the image data.

    Returns:
        None
    """
    image_data = []
    
    for root, _, files in os.walk(data_folder_path):
        for f in tqdm(files):
            if not f.endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_name = os.path.splitext(f)[0]
            label = 0 if "cat" in image_name else 1
            image_data.append([image_name, label])

    folder_name = os.path.basename(data_folder_path)
    csv_file_name = f"{folder_name}_label.csv"
    csv_file_path = os.path.join(data_folder_path, csv_file_name)

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(image_data)


