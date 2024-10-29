from datasets import load_dataset

import torch
from torch import nn
from torch import optim

from model import NetWork

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

test_data = load_dataset("ylecun/mnist")["test"]
test_data = test_data.map(lambda example: {"image": transform(example["image"]), 
                                             "label": torch.tensor(example["label"])})
test_data.set_format("torch") # data should be set torch again!

model = NetWork()
model.load_state_dict(torch.load('mnist.pth'))

cnt = 0 # True predicted labels

for item in test_data:
    image = item["image"]
    label = item["label"]
    output = model(image)
    predict_label = output.argmax(-1).item()
    if predict_label == label:
        cnt += 1
accuracy = float(cnt) / float(len(test_data))
print(f"Accuracy: {accuracy}")