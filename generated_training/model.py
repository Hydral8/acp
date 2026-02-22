import torch
import torch.nn as nn


class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb0 = nn.Embedding(30000, 512, padding_idx=0)
        self.ln4 = nn.LayerNorm(512)
        self.fc5 = nn.Linear(512, 64, bias=True)
        self.softmax6 = nn.Softmax(dim=-1)

    def forward(self, x):
        # input — expected shape: [512]
        h0 = self.emb0(x)  # [batch, seq] → [batch, seq, 512]
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
