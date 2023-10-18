import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    img = cv.imread("./input/test/test.jpg")
    img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
    # plt.imshow(img)
    # plt.show()

    model = torch.load("./save/best_1e-3_128.m")
    print(model)

    model_weights, conv_layers = [], []
    model_children = list(list(model.children())[0].children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
    print(f"Total convolution layers: {counter} layers")

    model = model.cpu()
    np.asarray(img, order="C")
    img = np.swapaxes(img, 0, 2)
    img = torch.Tensor(img)

    outputs, names = [], []
    for layer in conv_layers[0:]:
        img = layer(img)
        outputs.append(img)
        names.append(str(layer))
    print(len(outputs))
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    for i in range(len(processed)):
        plt.imshow(processed[i])
        plt.show()
