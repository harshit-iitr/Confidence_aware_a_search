import torch
from torch_model import ChrestienHeuristicNet

device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
print(f"Testing model on device: {device}")

model = ChrestienHeuristicNet(dim=10).to(device)
state = torch.randn(4, 5, 10, 10, device=device)
goal = torch.randn(4, 5, 10, 10, device=device)

out = model(state, goal)
print("Forward output shape:", out.shape)
loss = out.sum()
loss.backward()
print("Backward pass successful! Gradient norm:", sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None))
