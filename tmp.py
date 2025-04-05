import torch

# 假设使用 GPU 0
device = torch.device('cuda:0')

# 在训练前检查显存使用
print(f"初始显存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

# 创建一个张量
x = torch.randn(10000, 10000, device=device)

# 检查张量分配后的显存使用
print(f"分配张量后显存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

# 删除张量并清理
del x
torch.cuda.empty_cache()

# 检查清理后的显存使用
print(f"清理后显存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")