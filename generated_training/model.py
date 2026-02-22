import torch
import torch.nn as nn


class CompositeBlock0(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2048, 512), nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x

class CompositeBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2048, 512), nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_2 = CompositeBlock0()
        self.block_3 = CompositeBlock1()
        self.ln4 = nn.LayerNorm(512)
        self.fc5 = nn.Linear(512, 64, bias=True)
        self.softmax6 = nn.Softmax(dim=-1)

    def forward(self, x):
        # input — expected shape: [512]
        h2 = self.block_2(h1)
        h3 = self.block_3(h2)
        h4 = self.ln4(h3)
        h5 = self.fc5(h4)
        h6 = self.softmax6(h5)
        return h6


# ── Setup ──────────────────────────────────────────────────────────────────
model = GeneratedModel()
# optimizer not configured
# loss not configured


# ── Training Step ──────────────────────────────────────────────────────────
def train_step(batch_x, batch_y):
    model.train()
    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(batch_x, batch_y):
    model.eval()
    with torch.no_grad():
        output = model(batch_x)
        loss = criterion(output, batch_y)
    return loss.item()
