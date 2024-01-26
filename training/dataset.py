import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os
import random
from PIL import Image, ImageDraw

class RandomMask:
    def __init__(self, min_mask_size, max_mask_size):
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('Input must be a PIL image')

        draw = ImageDraw.Draw(img)
        width, height = img.size
        mask_width = random.randint(self.min_mask_size, self.max_mask_size)
        mask_height = random.randint(self.min_mask_size, self.max_mask_size)
        x1 = random.randint(0, width - mask_width)
        y1 = random.randint(0, height - mask_height)
        x2 = x1 + mask_width
        y2 = y1 + mask_height
        draw.rectangle([x1, y1, x2, y2], fill=0)
        return img

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    RandomMask(min_mask_size=10, max_mask_size=50),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_train_valid_loader(path):
    all_dataset = datasets.ImageFolder(path, transform=None)

    train_size = int(0.9 * len(all_dataset))
    valid_size = len(all_dataset) - train_size

    train_dataset, valid_dataset = random_split(all_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    train_dataset = TransformedSubset(train_dataset, transform=transform)
    valid_dataset = TransformedSubset(valid_dataset, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    return train_loader, valid_loader

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        img_path = self.imgs[index][0]
        _, filename = os.path.split(img_path)
        id_str, _ = os.path.splitext(filename)
        id_str = str(id_str)
        return img, label, id_str

def get_test_loader(path):
    test_dataset = ImageFolderWithPaths(path, transform=valid_transform)
    test_loader = DataLoader(test_dataset, shuffle=False)
    return test_loader
