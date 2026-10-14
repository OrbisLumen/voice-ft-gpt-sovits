"""在训练环境内运行：检查 CUDA 并实际执行卷积反向传播。"""
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA 不可用；检查 NVIDIA 驱动与 PyTorch CUDA 构建，不回退 CPU。')
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
layer = torch.nn.Conv1d(8, 16, 3).cuda()
x = torch.randn(2, 8, 4096, device='cuda', requires_grad=True)
layer(x).square().mean().backward()
torch.cuda.synchronize()
print('CUDA convolution backward: OK')
