Your Repo Name
# Convolutional Neural Network (CNN) Implementation & Visualizing Feature Maps of Models

This repository contains an implementation of a Convolutional Neural Network (CNN) using PyTorch for image processing tasks. The project includes functionalities for image augmentation, training, validating, and testing the model, as well as extracting and visualizing convolutional feature maps from a pre-trained model.

## Repository Structure
```
├── save/
│   └── mymodel.m
├── train.py
├── feat_vis.py
├── augmentation.py
├── data/
│   ├── train/
│   │   └── train_label.csv
│   └── test/
│       └── test_label.csv
├── requirements.txt
└── README.md
```

## Files and Directories
- `save/`: Directory to save the trained model.
- `data/`: Contains the test image for model evaluation. Please see the folder.
- `train.py`: Script for training, validating, and testing the CNN model.
- `feat_vis.py`: Script for loading and visualizing the convolution layer feature maps.
- `augmentation.py`: Script for data augmentation and CSV generation.
- `requirements.txt`: List of required Python packages.
- `README.md`: This readme file.


## Usage
### Training and Testing the Model
To train and test the CNN model, run the `train.py` script with the desired arguments:
```
python train.py --model <modeltype> --filename <filename> 
```
Flags:
- `--model`: Specifies the architecture of the model (e.g., CNN).
- `--filename`: Specifies the filename to save the trained model.
- `--data_path`: Specifies the path to the folder containing the training data.
- `--val_path`: Specifies the path to the folder containing the validation data.
- `--test_img`: Specifies the path to the test image.
- `--batch_size`: Specifies the batch size for training.
- `--lr`: Specifies the learning rate for the optimizer.
- `--epoch_num`: Specifies the number of epochs for training.

### Extracting and Visualizing Feature Maps
To extract and visualize the convolutional feature maps from a pre-trained model, run the `model.py` script:
```
python feat_vis.py --image_path <test_img> --model_path <model_path>
```

### Data Augmentation and CSV Generation
To perform data augmentation and/or generate a CSV file, run the `augmentation.py` script with the desired arguments:
```
python augmentation.py --action <action> --input_path <input_path> [--output_path <output_path>]
```
Flags:
- `--action`: Specifies the action to perform: 'augment' for data augmentation, 'gen_csv' for CSV generation, 'both' for both actions (default: 'both').
- `--input_path`: Specifies the path to the input data folder.
- `--output_path`: Specifies the path to the output data folder (only for augmentation).


## Scripts Overview
### `train.py`
Main Functions:
- `train(model, dataset)`: Trains the model on the given dataset.
- `validate(model, dataset)`: Validates the model on the given dataset.
- `test(model, target_img)`: Tests the model on a target image and predicts whether it's a dog or a cat.
- `get_acc(model, img, label)`: Computes the accuracy of the model.

### `feat_vis.py`
Main Functions:
- `read_image()`: Reads and displays the test image.
- `load_model()`: Loads the pre-trained model.
- `extract_conv(model)`: Extracts convolutional layers from the model.
- `transform_image(img)`: Transforms the image for model input.
- `process_feature_map(output)`: Processes feature maps from convolutional layers.
- `show_feature_maps(processed)`: Displays the processed feature maps.


## Acknowledgements
This project utilizes the following libraries and frameworks:
- PyTorch
- OpenCV
- Matplotlib
- Timm
