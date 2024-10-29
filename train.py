from datasets import load_dataset

import torch
from torch import nn
from torch import optim

from model import NetWork

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_data = load_dataset("ylecun/mnist")["train"]
train_data = train_data.map(lambda example: {"image": transform(example["image"]), 
                                            "label": torch.tensor(example["label"])})
train_data.set_format("torch") # data should be set torch again!

if __name__ == "__main__":

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

    model = NetWork()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    train_epoches = 2

    for epoch in range(train_epoches):
        for batch_idx, item in enumerate(train_loader):
            images, labels = item["image"], item["label"]
            scores = model(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{train_epoches} "
                        f"| Batch {batch_idx}/{len(train_loader)} "
                        f"| Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "mnist.pth")