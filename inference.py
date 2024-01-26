import torch
import csv
import os
from pathlib import Path

import dataset
from training.model import Resnet50Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# csv_path = 'predictions.csv'
predictions = []
classes = []

def get_classes(path):
    image_dir = Path(path)
    filepaths = list(image_dir.glob('**/*.jpg'))

    unique_classes = set()

    for filepath in filepaths:
        relative_path = filepath.relative_to(image_dir)
        class_name = relative_path.parts[0]
        unique_classes.add(class_name)

    unique_classes = sorted(list(unique_classes))

    return unique_classes

def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for img, _, img_path in test_loader:
            img= img.to(device)
            output = model(img)
            predicted = torch.argmax(output, dim=1)
            label = classes[predicted.item()]
            # print(img_path, label)
            predictions.append([img_path,label])

def write(predictions, csv_path):
    with open(csv_path, 'w',newline='') as f:
        header = ['id', 'label']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for id, pred in predictions:
            # print(id, "->",pred)
            writer.writerow({'id': id, 'label': pred})

if __name__ == "__main__":
    # using argparse to get the path of dataset
    import argparse
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--mp', default="110550107_weight.pth", type=str, help='model path')
    parser.add_argument('--sp', default='predictions.csv', type=str, help='csv path')
    args = parser.parse_args()
    classes = get_classes('data/train')

    model = Resnet50Model().to(device)
    print("ok")
    model.load_checkpoint(args.mp)

    test_loader = dataset.get_test_loader('data/test_head')
    print("length of test_loader:", len(test_loader))

    test_model(model, test_loader)
    print("finish testing")

    write(predictions, args.sp)