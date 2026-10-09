# src/model.py
import torch, torch.nn as nn

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    """Returns LOGITS (apply sigmoid for probabilities)."""
    def __init__(self, in_ch=3, base=16, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([block(in_ch, ch[0])] + [block(ch[i], ch[i+1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i+1], ch[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in range(depth)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(len(self.up))):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips.pop()], 1))
        return self.out(x)
