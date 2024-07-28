import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
import cv2 as cv
import numpy as np
from tqdm import tqdm
import argparse
from CNN import CNN
from Dataset import MyDataset
from util import transform_cv2mod
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, dataset, batch_size, LR):
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
    print(f"Train: avg_loss={avg_loss:.4f} avg_acc={avg_acc:.2f}")
    return avg_acc, avg_loss


@torch.no_grad()
def validate(model, dataset, batch_size):
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
    print(f"Validate: avg_loss={avg_loss:.4f} avg_acc={avg_acc:.2f}")
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


def main(model, filename, data_path, val_path, test_img, batch_size, LR, epoch_num):
    data = MyDataset(os.path.join(data_path, "train_label.csv"), data_path)
    val = MyDataset(os.path.join(val_path, "test_label.csv"), val_path)

    acc = 0
    for i in range(epoch_num):
        print(f"--- epoch {i} ---")
        train(model, data, batch_size, LR)
        temp, _ = validate(model, val, batch_size)
        if acc < temp:
            torch.save(model, f"save/{filename}")

    test(model, test_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a CNN model')
    parser.add_argument('-m', '--model', type=str, default='CNN', help='Model architecture')
    parser.add_argument('-f', '--filename', type=str, default='mymodel.m', help='Model filename')
    parser.add_argument('-d', '--data_path', type=str, default='./data/train', help='Path to the data folder')
    parser.add_argument('-v', '--val_path', type=str, default='./data/test', help='Path to the validation folder')
    parser.add_argument('-t', '--test_img', type=str, default='./data/test.jpg', help='Path to the test image')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('-e', '--epoch_num', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    model1 = CNN() if args.model == 'CNN' else timm.create_model(args.model, pretrained=True, num_classes=1)
    main(model1, args.filename, args.data_path, args.val_path, args.test_img, args.batch_size, args.LR, args.epoch_num)
