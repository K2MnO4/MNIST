from datasets import load_dataset

import time

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

cnt = 0 # True predicted labels
st_time = time.time()

with torch.no_grad():
    for item in test_data:
        image = item["image"].to(device)
        label = item["label"].to(device)
        image.to(device)
        label.to(device)
        output = model(image)
        predict_label = output.argmax(-1).item()
        if predict_label == label:
            cnt += 1
accuracy = float(cnt) / float(len(test_data))
ed_time = time.time()
print(f"Accuracy: {accuracy} Time: {ed_time - st_time}")