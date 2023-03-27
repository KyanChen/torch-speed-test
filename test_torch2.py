import torch
from torchvision.models import resnet50
import time

if __name__ == '__main__':
    torch_version = torch.__version__
    model = resnet50(weights=None)
    model.eval()
    max_try = 1000
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    x = torch.rand((1, 3, 512, 512), device=device)
    model = model.to(device=device)
    if torch_version == '2.0.0':
        model = torch.compile(model)

    for _ in range(50):
        with torch.no_grad():
            _ = model(x)

    t_time = 0
    for _ in range(max_try):
        with torch.no_grad():
            t_start = time.time()
            _ = model(x)
            t_time += time.time() - t_start
    print(f'{torch_version} fps = ', 1/(t_time / max_try))

# Resnet50
# Macos
# MPS torch1.13.1 37FPS
# MPS torch2.0.0 37FPS
# MPS torch2.0.0-compile 37FPS
# 1080Ti
# Linux 1.13.1+cu117-with-pypi-cudnn fps =  69.9
# Linux 2.0.0+cu117-with-pypi-cudnn fps =  69.3
# Linux 2.0.0+cu117-with-pypi-cudnn-compile fps =  失败
# A100
# Linux 1.13.1+cu117-with-pypi-cudnn fps =  148.6
# Linux 2.0.0+cu117-with-pypi-cudnn fps =  198.3
# Linux 2.0.0+cu117-with-pypi-cudnn fps =  278.4

# vit
# A100
# Linux 1.13.1+cu117-with-pypi-cudnn fps =  201.25
# Linux 2.0.0+cu117-with-pypi-cudnn fps =  216.72
# Linux 2.0.0+cu117-with-pypi-cudnn fps =  219.75

