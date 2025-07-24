import torch
import torch.nn as nn
import torch.nn.functional as F

def mixed_precision_accumulation():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 10, bias=False)
            self.ln = nn.LayerNorm(10)
            self.fc2 = nn.Linear(10, out_features, bias=False)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("after first", x.dtype)
        x = self.ln(x)
        print("after sec", x.dtype)
        x = self.fc2(x)
        print("after third", x.dtype)
        return x

def benchmarking_mixed_precision():
    device = "cuda"
    model = ToyModel(100, 5).to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        input = torch.rand(100).to(device)
        print(input, input.dtype)
        out = model(input).to("cpu")
        y = torch.tensor(2)
        loss = F.cross_entropy(out, y)
        print(out, out.dtype)
        print(f"loss, {loss.item()} and dtype: {loss.dtype}")

        loss.backward()
        for p in model.parameters():
            print("p", p)
            print("grad", p.grad, p.grad.dtype)
        
if __name__ == "__main__":
    # mixed_precision_accumulation()
    benchmarking_mixed_precision()