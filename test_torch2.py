import torch
from torchvision.models import resnet50
import time

if __name__ == '__main__':
    torch_version = torch.__version__
    model = resnet50(weights=None)
    model.eval()
    max_try = 1
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    x = torch.rand((1, 3, 512, 512), device=device)
    model = model.to(device=device)
    if torch_version == '2.0.0':
        model = torch.compile(model)
    t_time = 0
    for _ in range(max_try):
        with torch.no_grad():
            t_start = time.time()
            _ = model(x)
            t_time += time.time() - t_start
    print(f'{torch_version} fps = ', 1/(t_time / max_try))

