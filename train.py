import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
import cv2 as cv
import numpy as np
from tqdm import tqdm

from CNN import CNN
from Dataset import MyDataset
from util import transform_cv2mod
import argparse

PATH = "./input/"
LR = 3e-4
batch_size = 32
epoch_num = 10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = MyDataset(PATH + "train_label.csv", PATH + "data/data/")
val = MyDataset(PATH + "test_label.csv", PATH + "test/test/")


def train(model, dataset):
    model.to(device).train()
    batch_count, total_acc, total_loss = 0, 0, 0
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.MSELoss()
    optim_func = optim.Adam(model.parameters(), lr=LR)

    for img, label in tqdm(train_loader, desc="Train: ", unit="batch(es)", unit_scale=True):
        img = img.float().to(device)
        label = label.float().to(device)

        y = model(img)
        optim_func.zero_grad()
        loss = loss_func(y, label.unsqueeze(-1))

        loss.backward()
        optim_func.step()

        total_loss += loss.detach().item()
        total_acc += get_acc(model, img, label)
        batch_count += 1

    avg_loss, avg_acc = total_loss / batch_count, total_acc / batch_count
    print(f"Train: avg_loss={avg_loss} avg_acc={avg_acc}")
    return avg_acc, avg_loss


@torch.no_grad()
def validate(model, dataset):
    model.eval()
    batch_count, total_acc, total_loss = 0, 0, 0
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.MSELoss()

    for img, label in tqdm(train_loader, desc="Val: ", unit="batch(es)", unit_scale=True):
        img = img.type(torch.float32).to(device)
        label = label.type(torch.float32).to(device)

        y = model(img)
        loss = loss_func(y, label.unsqueeze(-1))

        total_loss += loss.detach().item()
        total_acc += get_acc(model, img, label)
        batch_count += 1

    avg_loss, avg_acc = total_loss / batch_count, total_acc / batch_count
    print(f"Validate: avg_loss={avg_loss} avg_acc={avg_acc}")
    return avg_acc, avg_loss


@torch.no_grad()
def test(model, target_img):
    img = cv.imread(target_img)
    img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)

    model.cpu()
    model.eval()
    img = torch.Tensor(transform_cv2mod(img)).unsqueeze(0)

    y = "dog" if model(img)[0][0] > 0.5 else "cat"
    print(y)


def get_acc(model, img, label):
    model.eval()
    output = model(img)
    correct = torch.sum((output > 0.5).squeeze(-1) == label)
    total = img.size()[0]
    return correct / total


def main(model, filename):
    acc = 0
    for i in range(epoch_num):
        print(f"--- epoch {i} ---")
        train(model, data)
        temp, _ = validate(model, val)
        if acc < temp:
            torch.save(model, f"save/{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a CNN model')
    parser.add_argument('--model', type=str, default='CNN', help='Model architecture')
    parser.add_argument('--filename', type=str, default='mymodel.m', help='Model filename')
    args = parser.parse_args()

    model1 = CNN() if args.model == 'CNN' else timm.create_model(args.model, pretrained=True, num_classes=1)
    main(model1, args.filename)
    test(model1, "./input/test/test.jpg")
