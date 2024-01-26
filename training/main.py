import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import dataset
from model import Resnet50Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []
checkpoint_path = 'final/checkpoint.pth'

def correctes(output, label):
    _, predicted = torch.max(output, 1)
    correct = (predicted == label).sum().item()
    return correct

def plot(path, epochs, lr):
    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Resnet50: Loss epoch={epochs} lr={lr}')
    plt.legend()
    plt.savefig(path + f"Resnet50_ep{epochs}_lr{lr}_loss.png")
    plt.close()

    plt.plot(train_accs, label='train acc')
    plt.plot(valid_accs, label='valid acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Resnet50: Accuracy epoch={epochs} lr={lr}')
    plt.legend()
    plt.savefig(path + f"Resnet50_ep{epochs}_lr{lr}_acc.png")
    plt.close()

def train_model(model, train_loader, valid_loader, epochs, optimizer, scheduler, criterion, checkpoint_path, best_valid_loss=float('inf')):

    train_len = len(train_loader.dataset)
    valid_len = len(valid_loader.dataset)
    # train loop
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0

        model.train()
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', dynamic_ncols=True)
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc = correctes(output, label)
            train_loss += loss.item()
            train_acc += acc
            # train_loader.set_postfix({'Train Loss': loss.item(), 'Train Acc': acc/len(label)})

        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/train_len)


        # validation loop
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_acc = 0.0
            for img, label in valid_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += correctes(output, label)

            valid_losses.append(valid_loss/len(valid_loader))
            valid_accs.append(valid_acc/valid_len)
        
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            torch.save(model.state_dict(), f"Resnet50/{valid_loss}.pth")
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}'
            .format(epoch+1, epochs, train_losses[-1], valid_losses[-1], train_accs[-1], valid_accs[-1]))
        
    return train_losses, valid_losses, train_accs, valid_accs

if __name__ == "__main__":
    # using argparse to get the path of dataset
    import argparse
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dp', default="../data/train", type=str, help='dataset path')
    parser.add_argument('--fp', default=None, type=str, help='finetune path')
    parser.add_argument('--name', default=None, type=str, help='model name')
    parser.add_argument('--pp', default="Plots", type=str, help='plot path')
    args = parser.parse_args()

    train_loader, valid_loader = dataset.get_train_valid_loader('../data/train')
    print('data loaded')

    learning_rate = 0.0003
    model = Resnet50Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    epochs = 10

    if args.fp is not None:
        model.load_checkpoint(args.fp)
    
    os.makedirs("Resnet50", exist_ok=True)
    os.makedirs(args.pp, exist_ok=True)

    print('start training')
    train_losses, valid_losses, train_accs, valid_accs = train_model(model, train_loader, valid_loader, epochs, optimizer, scheduler, criterion, checkpoint_path)

    torch.save(model.state_dict(), f"Resnet50/Resnet50_sclr{learning_rate}.pth")
    print(f"model saved to Resnet50_sclr{learning_rate}.pth")

    # plot loss and accuracy
    plot(args.pp, epochs, learning_rate)