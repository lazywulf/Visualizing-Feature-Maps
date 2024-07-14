import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
from util import transform_cv2mod


def read_image():
    img = cv.imread("./input/test/test.jpg")
    img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
    plt.imshow(img)
    plt.show()
    return img

def load_model():
    model = torch.load("./save/mymodel.m")
    model = model.cpu()
    return model

def extract_conv(model):
    conv_layers = {}
    model_children = list(list(model.children())[0].children())
    counter = 0
    for i, layer in enumerate(model_children):
        if isinstance(layer, nn.Conv2d):
            counter += 1
            layer_name = f"ConvLayer_{counter}"
            conv_layers[layer_name] = layer
    print(f"Total convolution layers: {counter} layers")
    return conv_layers

def transform_image(img):
    img = torch.Tensor(transform_cv2mod(img))
    return img

def process_feature_map(output):
    processed = {}
    for name, feat_map in output.items():
        feat_map = feat_map.squeeze(0)
        gray_scale = torch.sum(feat_map, 0)
        gray_scale = gray_scale / feat_map.shape[0]
        processed[name] = gray_scale.data.cpu().numpy()
    return processed

def show_feature_maps(processed):
    for name, feat_map in processed.items():
        plt.imshow(feat_map)
        plt.colorbar()
        plt.title(name)
        plt.show()

def main():
    img = read_image()
    img = transform_image(img) 
    model = load_model()
    conv_layers = extract_conv(model)

    outputs = {}
    for name, layer in conv_layers.items():
        img = layer(img)
        outputs[name] = img

    processed = process_feature_map(outputs)
    show_feature_maps(processed)

if __name__ == "__main__":
    main()
