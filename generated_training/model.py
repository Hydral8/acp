import torch
import torch.nn as nn


class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb0 = nn.Embedding(30000, 512, padding_idx=0)
        self.emb1 = nn.Embedding(30000, 512, padding_idx=0)
        self.drop3 = nn.Dropout(p=0.5)
        self.ln4 = nn.LayerNorm(512)
        self.ln7 = nn.LayerNorm(512)
        self.ln10 = nn.LayerNorm(512)
        self.ln13 = nn.LayerNorm(512)
        self.ln16 = nn.LayerNorm(512)
        self.fc17 = nn.Linear(512, 64, bias=True)

    def forward(self, x):
        # input — expected shape: [512]
        h0 = self.emb0(x)  # [batch, seq] → [batch, seq, 512]
        h1 = self.emb1(x)  # [batch, seq] → [batch, seq, 512]
        h3 = self.drop3(h2)
        h4 = self.ln4(h3)
        h7 = self.ln7(h6)
        h10 = self.ln10(h9)
        h13 = self.ln13(h12)
        h16 = self.ln16(h15)
        h17 = self.fc17(h16)
        return h18


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
