import csv
import os
import cv2 as cv

cat_dir = "PetImages\\Preprocess\\Cat\\"
dog_dir = "PetImages\\Preprocess\\Dog\\"
cat_target = "PetImages\\Resized\\Cat\\"
dog_target = "PetImages\\Resized\\Dog\\"


def resize():
    for img_file in os.listdir(cat_dir):
        try:
            img = cv.imread(cat_dir + img_file)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
            cv.imwrite(cat_target + img_file, img=img)
        except:
            print(img_file)

    for img_file in os.listdir(dog_dir):
        try:
            img = cv.imread(dog_dir + img_file)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
            cv.imwrite(dog_target + img_file, img=img)
        except:
            print(img_file)


def rename():
    for index, img_file in enumerate(os.listdir(cat_target + "test\\")):
        os.rename(cat_target + "test\\" + img_file, cat_target + "test\\" + f"cat_test{index}.jpg")
    print("1")
    for index, img_file in enumerate(os.listdir(dog_target + "test\\")):
        os.rename(dog_target + "test\\" + img_file, dog_target + "test\\" + f"dog_test{index}.jpg")
    print("1")
    for index, img_file in enumerate(os.listdir(cat_target)):
        os.rename(cat_target + img_file, cat_target + f"cat{index}.jpg")
    print("1")
    for index, img_file in enumerate(os.listdir(dog_target)):
        os.rename(dog_target + img_file, dog_target + f"dog{index}.jpg")
    print("1")


def csv_log():
    with open("train_label.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(7917):
            writer.writerow([f"cat{i}.jpg", 0])
        for i in range(10471):
            writer.writerow([f"dog{i}.jpg", 1])
    with open("test_label.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(1999):
            writer.writerow([f"cat_test{i}.jpg", 0])
        for i in range(1999):
            writer.writerow([f"dog_test{i}.jpg", 1])


if __name__ == "__main__":
    csv_log()