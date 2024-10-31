import torch
from torch import nn


class NetWork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # define first liner layer
        self.hidden_layer = nn.Linear(784, 256)
        # define second liner layer
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28 * 28)
        x = self.hidden_layer(x)
        x = torch.relu(x) # Use relu
        return self.fc(x)

def print_parameters(model: NetWork):
    param_num = 0
    for name, layer in model.named_children():
        print(f"{name} parameters: ")
        for p in layer.parameters():
            print(f"\t {p.shape} has {p.numel()} parameters")
            param_num += p.numel()
    print(f"The total number of trainable parameters: {param_num}")

def print_forward_process(model: NetWork, x: torch.Tensor):
    print(f"example shape: {x.shape}")
    x = x.view(-1, 28 * 28)
    print(f"shape after flatting: {x.shape}")
    x = model.hidden_layer(x)
    print(f"shape after hidden layer: {x.shape}")
    x = torch.relu(x)
    print(f"shape after relu: {x.shape}")
    x = model.fc(x)
    print(f"shape after linear layer: {x.shape}")



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = NetWork()
    model.to(device)
    # print(model.state_dict())
    # print_parameters(model)
    x = torch.zeros([5, 28, 28]).to(device)
    print_forward_process(model, x)