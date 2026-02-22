"""train.py — training loop with Weights & Biases tracking"""
import os
import torch
import torch.nn as nn
import wandb
from model import Model
from data import get_dataloaders

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LR            = 0.0003
BATCH         = 32
EPOCHS        = 3
SAVE_PATH     = "checkpoint_best.pt"
WANDB_PROJECT = "simple-gpt-training"
WANDB_ENTITY  = None   # set to your W&B username/team, or None for default


def train():
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "lr":          LR,
            "batch_size":  BATCH,
            "epochs":      EPOCHS,
            "optimizer":   "AdamW",
            "loss":        "CrossEntropyLoss",
            "device":      str(DEVICE),
        },
    )

    # ── Print run URL — orchestration layer captures this line ────────────
    print(f"WANDB_RUN_URL: {run.get_url()}", flush=True)

    model     = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
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
            out   = model(x)
            loss  = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validation ────────────────────────────────────────────────────
        log_dict = {"epoch": epoch, "train/loss": train_loss}
        val_info = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    val_loss += criterion(model(x), y).item()
            val_loss /= len(val_loader)
            log_dict["val/loss"] = val_loss
            val_info = f"  val={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                wandb.save(SAVE_PATH)
                val_info += " ✓saved"

        wandb.log(log_dict)
        print(f"Epoch {epoch:03d}/{EPOCHS}  train={train_loss:.4f}{val_info}")

    torch.save(model.state_dict(), "model_final.pt")
    wandb.save("model_final.pt")
    wandb.finish()
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
