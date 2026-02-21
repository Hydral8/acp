"""train.py — training loop"""
import torch
import torch.nn as nn
from model import Model
from data import get_dataloaders

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LR        = 0.001
BATCH     = 32
EPOCHS    = 10
SAVE_PATH = "checkpoint_best.pt"


def train():
    model     = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = get_dataloaders(batch_size=BATCH)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validation ────────────────────────────────────────────────────
        val_info = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    val_loss += criterion(model(x), y).item()
            val_loss /= len(val_loader)
            val_info = f"  val={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                val_info += " ✓saved"

        print(f"Epoch {epoch:03d}/{EPOCHS}  train={train_loss:.4f}{val_info}")

    torch.save(model.state_dict(), "model_final.pt")
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
