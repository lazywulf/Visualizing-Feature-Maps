import os
import argparse
from util import rand_aug, generate_csv

def main():
    parser = argparse.ArgumentParser(description="Data Augmentation and CSV Generation")
    parser.add_argument('--action', type=str, default='both', choices=['augment', 'gen_csv', 'both'], help="Action to perform: 'augment' for data augmentation, 'gen_csv' for CSV generation, 'both' for both actions")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input data folder")
    parser.add_argument('--output_path', type=str, help="Path to the output data folder (only for augmentation)")

    args = parser.parse_args()

    if args.action == 'augment':
        rand_aug(args.input_path, args.output_path)
    elif args.action == 'gen_csv':
        generate_csv(args.input_path)
    elif args.action == 'both':
        rand_aug(args.input_path, args.output_path)
        generate_csv(args.output_path if args.output_path else args.input_path)


if __name__ == "__main__":
    main()